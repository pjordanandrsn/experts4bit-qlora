"""Replay a routing trace through residency policies at fixed budgets: hit rate and staged bytes. CPU only, no timings.

A residency tier holds `budget` (layer, expert) ROWS. Every pass of every layer touches the distinct experts its tokens
were routed to; a touch of a row that is not resident is a miss, and a miss stages one expert row's bytes.

Training access order, per micro-batch (gradient checkpointing; this is how the step runs, not a modelling choice):
  forward    layers 0 .. L-1, each layer's distinct forward set;
  backward   layers L-1 .. 0, each layer's recompute set, then its dgrad set. The dgrad runs on the recompute's routing,
             so the dgrad set IS the recompute set (by construction; it is not separately hooked).
Decode access order (the #1469 traces): per decoded token, layers 0 .. L-1, the routed set.

Policies (each at every budget):
  profile   a static resident set: the top-`budget` rows by forward frequency in the FIT steps, scored on the rest
            (held out); never changes;
  lfu       least-frequently-used eviction, least-recently-used among ties (the shipped ColdTier rule);
  lru       least-recently-used eviction;
  belady    evict the row whose next use is farthest (the offline upper bound).
Every policy is scored on the same window (the steps after the fit window); the dynamic policies run through the fit
window first, so they are scored warm.

Controls: `--null` replaces every micro-batch's per-layer set with a random set of the same size (seeded), keeping the
forward set equal to the recompute set, so within-step reuse stays and cross-micro-batch locality is destroyed.

    python replay.py --trace trace.npz --fit-steps 5 --out replay.json
    python replay.py --decode qwen3_prose.jsonl --fit-steps 128 --out decode.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import OrderedDict, defaultdict

import numpy as np

KEY = re.compile(r"^s(\d+)_mb(\d+)_(fwd|recompute)_L(\d+)$")
BUDGETS = (128, 384, 768, 1024, 3216, 6144)
POLICIES = ("profile", "lfu", "lru", "belady")


def expert_row_bytes(hidden: int, moe_intermediate: int, block: int = 64, absmax_bytes: int = 4) -> int:
    """One expert's packed NF4 bytes (gate, up, down: 3 x hidden x intermediate weights at 4 bits) plus one absmax of
    `absmax_bytes` per `block` weights (fp32 per 64 by default, the gnf4 grouped layout)."""
    n = 3 * hidden * moe_intermediate
    return n // 2 + (n // block) * absmax_bytes


# -- building the access sequence --------------------------------------------------------------------------------
def load_train_trace(path: str) -> dict[tuple[int, int], dict[str, dict[int, np.ndarray]]]:
    """{(step, mb): {"fwd": {layer: ids}, "recompute": {...}}} for the micro-batches that have BOTH passes (training
    micro-batches); forward-only passes are dropped here and counted by the recorder's own summary."""
    data = np.load(path)
    mbs: dict[tuple[int, int], dict[str, dict[int, np.ndarray]]] = defaultdict(lambda: {"fwd": {}, "recompute": {}})
    for k in data.files:
        m = KEY.match(k)
        if m:
            s, mb, p, layer = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            mbs[(s, mb)][p][layer] = data[k]
    return {k: v for k, v in mbs.items() if v["fwd"] and v["recompute"]}


def train_sequence(mbs, null_seed: int | None = None, n_experts: int = 128):
    """[(step, row)] in execution order; row = (layer, expert). `null_seed` gives the shuffled-routing null."""
    rng = np.random.default_rng(null_seed) if null_seed is not None else None
    seq = []
    for (step, mb) in sorted(mbs):
        fwd = {layer: np.unique(ids) for layer, ids in mbs[(step, mb)]["fwd"].items()}
        rec = {layer: np.unique(ids) for layer, ids in mbs[(step, mb)]["recompute"].items()}
        if rng is not None:                       # same size per layer, random members; recompute keeps the forward's set
            fwd = {layer: np.sort(rng.choice(n_experts, size=len(s), replace=False)) for layer, s in fwd.items()}
            rec = {layer: fwd[layer] for layer in rec}
        for layer in sorted(fwd):
            seq += [(step, (layer, int(e))) for e in fwd[layer]]
        for layer in sorted(rec, reverse=True):
            seq += [(step, (layer, int(e))) for e in rec[layer]]      # the recompute
            seq += [(step, (layer, int(e))) for e in rec[layer]]      # the dgrad: the recompute's routing
    return seq


def decode_sequence(path: str):
    """[(decode step, row)] from a #1469 decode trace (JSONL: a meta line, then {"step", "routed": {layer: ids}})."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    seq = []
    for r in rows[1:]:
        for layer in sorted(r["routed"], key=int):
            seq += [(int(r["step"]), (int(layer), int(e))) for e in sorted(set(r["routed"][layer]))]
    return seq


# -- policies ----------------------------------------------------------------------------------------------------
def simulate(seq, budget: int, policy: str, fit_steps: int):
    """Hits and touches over the scored window (steps >= fit_steps)."""
    hits = touches = 0
    if policy == "profile":
        freq: dict = defaultdict(int)
        for step, row in seq:
            if step < fit_steps:
                freq[row] += 1
        resident = {r for r, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]}
        for step, row in seq:
            if step >= fit_steps:
                touches += 1
                hits += row in resident
        return hits, touches

    if policy == "belady":
        nxt = [0] * len(seq)                      # index of each touch's next use of the same row
        last: dict = {}
        for i in range(len(seq) - 1, -1, -1):
            row = seq[i][1]
            nxt[i] = last.get(row, len(seq) + i)
            last[row] = i
        resident: dict = {}                       # row -> its next use
        heap: list = []                           # (-next use, row), lazily invalidated
        for i, (step, row) in enumerate(seq):
            hit = row in resident
            if step >= fit_steps:
                touches += 1
                hits += hit
            if not hit and len(resident) >= budget:
                while heap:
                    nu, victim = heapq.heappop(heap)
                    if resident.get(victim) == -nu:
                        del resident[victim]
                        break
            resident[row] = nxt[i]
            heapq.heappush(heap, (-nxt[i], row))
        return hits, touches

    if policy == "lru":
        cache: OrderedDict = OrderedDict()
        for step, row in seq:
            hit = row in cache
            if step >= fit_steps:
                touches += 1
                hits += hit
            if hit:
                cache.move_to_end(row)
            else:
                if len(cache) >= budget:
                    cache.popitem(last=False)
                cache[row] = True
        return hits, touches

    if policy == "lfu":
        count: dict = defaultdict(int)            # frequency over the whole history (ColdTier keeps counts)
        stamp: dict = {}                          # last use, the tie-break
        heap: list = []                           # (count, last use, row), lazily invalidated
        resident: set = set()
        for t, (step, row) in enumerate(seq):
            hit = row in resident
            if step >= fit_steps:
                touches += 1
                hits += hit
            count[row] += 1
            stamp[row] = t
            if not hit:
                if len(resident) >= budget:
                    while heap:
                        c, s, victim = heapq.heappop(heap)
                        if victim in resident and count[victim] == c and stamp[victim] == s:
                            resident.discard(victim)
                            break
                resident.add(row)
            heapq.heappush(heap, (count[row], t, row))
        return hits, touches

    raise ValueError(f"unknown policy {policy!r}")


def census(seq, fit_steps: int) -> dict:
    """Distinct experts per (step, layer) and the step-to-step Jaccard of each layer's set."""
    sets: dict = defaultdict(set)
    for step, (layer, e) in seq:
        sets[(step, layer)].add(e)
    steps = sorted({s for s, _ in sets})
    layers = sorted({layer for _, layer in sets})
    distinct = [len(sets[(s, layer)]) for s in steps for layer in layers if (s, layer) in sets]
    jac = []
    for a, b in zip(steps, steps[1:]):
        for layer in layers:
            x, y = sets.get((a, layer), set()), sets.get((b, layer), set())
            if x | y:
                jac.append(len(x & y) / len(x | y))
    return {"steps": len(steps), "layers": len(layers),
            "distinct_per_step_layer": {"mean": float(np.mean(distinct)) if distinct else None,
                                        "min": min(distinct) if distinct else None,
                                        "max": max(distinct) if distinct else None},
            "jaccard_step_to_step": {"mean": float(np.mean(jac)) if jac else None,
                                     "min": float(min(jac)) if jac else None}}


def replay(seq, fit_steps: int, row_bytes: int, budgets=BUDGETS, policies=POLICIES) -> dict:
    scored_steps = len({s for s, _ in seq if s >= fit_steps})
    out = {"census": census(seq, fit_steps), "row_bytes": row_bytes, "fit_steps": fit_steps,
           "scored_steps": scored_steps, "touches_scored": sum(1 for s, _ in seq if s >= fit_steps), "cells": []}
    for b in budgets:
        for p in policies:
            h, t = simulate(seq, b, p, fit_steps)
            out["cells"].append({"budget": b, "policy": p, "hits": h, "touches": t,
                                 "hit_rate": h / t if t else None,
                                 "staged_bytes_per_step": (t - h) * row_bytes / scored_steps if scored_steps else None})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="a training trace .npz from run_capture.py")
    src.add_argument("--decode", help="a #1469 decode trace .jsonl")
    ap.add_argument("--fit-steps", type=int, default=5)
    ap.add_argument("--null-seed", type=int, default=None, help="replay the shuffled-routing null with this seed")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--moe-intermediate", type=int, default=768)
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.trace:
        seq = train_sequence(load_train_trace(a.trace), null_seed=a.null_seed, n_experts=a.n_experts)
    else:
        seq = decode_sequence(a.decode)
    res = replay(seq, a.fit_steps, expert_row_bytes(a.hidden, a.moe_intermediate))
    res["source"] = a.trace or a.decode
    res["null_seed"] = a.null_seed
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({c["policy"] + "@" + str(c["budget"]): round(c["hit_rate"], 4) for c in res["cells"]
                      if c["hit_rate"] is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
