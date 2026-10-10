"""Replay a routing trace through residency policies at fixed budgets: hit rate and staged bytes. CPU only, no timings.

A residency tier holds `budget` (layer, expert) ROWS. Every pass of every layer touches the distinct experts its tokens
were routed to; a touch of a row that is not resident is a miss, and a miss stages one expert row's bytes.

Training access order, per micro-batch (gradient checkpointing; this is how the step runs, not a modelling choice):
  fwd        layers 0 .. L-1, each layer's distinct forward set;
  recompute  layers L-1 .. 0, each layer's recompute set, then
  dgrad      that layer's dgrad set. The dgrad runs on the recompute's routing, so the dgrad set IS the recompute set (by
             construction; it is not separately hooked). Every dgrad touch follows its layer's recompute directly, so
             the dgrad pass is near-guaranteed hits: hit rates are reported PER PASS as well as pooled.
Decode access order (the #1469 traces): per decoded token, layers 0 .. L-1, the routed set; pass `decode`.

Policies (each at every budget):
  profile   a static resident set: the top-`budget` rows by FORWARD frequency in the FIT steps, scored on the rest
            (held out); never changes;
  lfu       least-frequently-used eviction, least-recently-used among ties (the shipped ColdTier rule);
  lru       least-recently-used eviction;
  belady    evict the row whose next use is farthest (the offline upper bound).
Every policy is scored on the same window (the steps after the fit window); the dynamic policies run through the fit
window first, so they are scored warm.

A training trace is refused unless it is complete: its capture verdict is OK, and every step holds exactly --expect-mb
micro-batches, each with every layer in both passes, and every forward routes as its recompute (re-checked here from
the file, not trusted from the meta). A capture that ended early (an OOM in a later step) can be scored on its first N
optimizer steps with --complete-steps N --truncation-reason TEXT: the same checks run on exactly the kept steps, a
fault inside the cut still refuses, and the output's trace_info records the capture's verdict, the planned steps and
the cut.

Controls: `--null-seed` replaces every micro-batch's per-layer set with a random set of the same size (seeded), keeping
the forward set for the recompute and the dgrad, so within-step reuse stays and cross-micro-batch locality is destroyed.

    python replay.py --trace trace.npz --expect-mb 8 --fit-steps 5 --out replay.json
    python replay.py --decode qwen3_prose.jsonl --fit-steps 128 --out decode.json
"""
from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

KEY = re.compile(r"^s(\d+)_mb(\d+)_(fwd|recompute)_L(\d+)$")
BUDGETS = (128, 384, 768, 1024, 3216, 6144)
POLICIES = ("profile", "lfu", "lru", "belady")
TRAIN_PASSES = ("fwd", "recompute", "dgrad")

#: The per-expert layout OFFLOAD_EXPERTS stages, with where each fact is read (experts4bit-qlora, this commit):
#:   packed NF4 `gate_up_proj` / `down_proj` (uint8) and `gate_up_absmax` / `down_absmax` (float32 buffers):
#:     experts4bit_qlora/engines/offload.py:4-5, 251;
#:   double-quantized absmax (E4B_ABSMAX_DQ=1) is REFUSED by expert offload, so the absmax is fp32:
#:     experts4bit_qlora/engines/offload.py:737-743;
#:   blocksize 64: the streaming loader's default, experts4bit_qlora/loader.py:1177; bench/tc1/tc1_arm.py passes none and
#:     records the regime it loaded (`nf4/64`) in its receipt.
#: OFFLOAD_EXPERTS itself stages a layer's WHOLE expert stack; the row figure here prices a row-granular tier (one expert
#: of the same layout), which is what a residency budget in rows means.
LAYOUT = {"format": "nf4", "blocksize": 64, "absmax_dtype": "float32", "absmax_bytes": 4, "double_quant": False,
          "sources": ["experts4bit_qlora/engines/offload.py:4-5,251", "experts4bit_qlora/engines/offload.py:737-743",
                      "experts4bit_qlora/loader.py:1177"]}


def expert_row_bytes(hidden: int, moe_intermediate: int, block: int = 64, absmax_bytes: int = 4) -> int:
    """One expert's packed NF4 bytes (gate, up, down: 3 x hidden x intermediate weights at 4 bits) plus one absmax of
    `absmax_bytes` per `block` weights (the layout in LAYOUT)."""
    n = 3 * hidden * moe_intermediate
    return n // 2 + (n // block) * absmax_bytes


def _recorder():
    path = Path(__file__).resolve().parent / "routing_recorder.py"
    spec = importlib.util.spec_from_file_location("routing_recorder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class IncompleteTrace(ValueError):
    """A training trace that does not contain what it claims: refused, never replayed."""


# -- building the access sequence --------------------------------------------------------------------------------
def _planned_steps(meta: dict):
    args = meta.get("harness_args") or []
    return int(args[args.index("--steps") + 1]) if "--steps" in args else None


def load_train_trace_info(path: str, expect_mb: int, complete_steps: int | None = None, reason: str | None = None):
    """(micro-batches, info) after the completeness guard. Raises IncompleteTrace.

    `complete_steps=N` scores only the first N optimizer steps of a capture that ended early (an OOM in a later step):
    the capture's own verdict may then be VOID, but the guard and the forward == recompute check are re-run on exactly
    the kept steps, and anything wrong inside the cut still refuses. N cannot exceed the optimizer steps the capture
    completed, the capture must carry its meta (the provenance of the cut), and `reason` must say why. `info` records
    the capture's verdict and reasons, the planned steps (`truncated_from`), N and the reason."""
    meta_path = path + ".meta.json"
    meta, summary = {}, {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            doc = json.load(f)
        meta, summary = doc.get("meta", {}), doc.get("summary", {})
    verdict = summary.get("verdict")
    info = {"capture_verdict": verdict, "complete_steps": complete_steps}
    if complete_steps is None:
        if os.path.exists(meta_path) and verdict != "OK":
            raise IncompleteTrace(f"capture verdict {verdict!r}: {summary.get('reasons', [])[:5]}")
    else:
        if not summary:
            raise IncompleteTrace("--complete-steps needs the capture's meta: the cut must be traceable to the capture")
        if not reason:
            raise IncompleteTrace("--complete-steps needs a stated reason for the cut")
        if not 0 < complete_steps <= int(summary.get("steps", 0)):
            raise IncompleteTrace(f"--complete-steps {complete_steps}: the capture completed "
                                  f"{summary.get('steps')} optimizer steps")
        info.update({"capture_reasons": summary.get("reasons", []), "truncated_from": _planned_steps(meta),
                     "reason": reason})
    data = np.load(path)
    records, layers, steps = {}, set(), 0
    for k in data.files:
        m = KEY.match(k)
        if not m:
            continue
        s, mb, p, layer = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        layers.add(layer)
        if complete_steps is not None and s >= complete_steps:
            continue
        records[(s, mb, p, layer)] = data[k]
        steps = max(steps, s + 1)
    if complete_steps is not None:
        steps = complete_steps
    reasons = _recorder().completeness_reasons(records, sorted(layers), steps, expect_mb)
    differ = [(s, mb, layer) for (s, mb, p, layer), ids in records.items()
              if p == "fwd" and not np.array_equal(ids, records.get((s, mb, "recompute", layer)))]
    if differ:
        reasons.append(f"{len(differ)} (step, micro-batch, layer) whose recompute routed differently from its forward, "
                       f"first {sorted(differ)[:3]}")
    if reasons:
        raise IncompleteTrace(f"refused: {reasons[:5]}")
    mbs: dict = defaultdict(lambda: {"fwd": {}, "recompute": {}})
    for (s, mb, p, layer), ids in records.items():
        mbs[(s, mb)][p][layer] = ids
    return dict(mbs), info


def load_train_trace(path: str, expect_mb: int, complete_steps: int | None = None, reason: str | None = None):
    """{(step, mb): {"fwd": {layer: ids}, "recompute": {...}}}, after the completeness guard. Raises IncompleteTrace."""
    return load_train_trace_info(path, expect_mb, complete_steps, reason)[0]


def train_sequence(mbs, null_seed: int | None = None, n_experts: int = 128):
    """[(step, pass, row)] in execution order; row = (layer, expert). `null_seed` gives the shuffled-routing null."""
    rng = np.random.default_rng(null_seed) if null_seed is not None else None
    seq = []
    for (step, mb) in sorted(mbs):
        fwd = {layer: np.unique(ids) for layer, ids in mbs[(step, mb)]["fwd"].items()}
        rec = {layer: np.unique(ids) for layer, ids in mbs[(step, mb)]["recompute"].items()}
        if rng is not None:                       # same size per layer, random members; recompute keeps the forward's set
            fwd = {layer: np.sort(rng.choice(n_experts, size=len(s), replace=False)) for layer, s in fwd.items()}
            rec = {layer: fwd[layer] for layer in rec}
        for layer in sorted(fwd):
            seq += [(step, "fwd", (layer, int(e))) for e in fwd[layer]]
        for layer in sorted(rec, reverse=True):
            seq += [(step, "recompute", (layer, int(e))) for e in rec[layer]]
            seq += [(step, "dgrad", (layer, int(e))) for e in rec[layer]]      # the recompute's routing
    return seq


def decode_sequence(path: str):
    """[(decode step, "decode", row)] from a #1469 decode trace (JSONL: a meta line, then {"step", "routed"})."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    seq = []
    for r in rows[1:]:
        for layer in sorted(r["routed"], key=int):
            seq += [(int(r["step"]), "decode", (int(layer), int(e))) for e in sorted(set(r["routed"][layer]))]
    return seq


# -- policies ----------------------------------------------------------------------------------------------------
def simulate(seq, budget: int, policy: str, fit_steps: int) -> dict[str, tuple[int, int]]:
    """{pass: (hits, touches)} over the scored window (steps >= fit_steps), plus "pooled"."""
    hits: dict = defaultdict(int)
    touches: dict = defaultdict(int)

    def score(step, p, hit):
        if step >= fit_steps:
            touches[p] += 1
            hits[p] += int(hit)

    if policy == "profile":
        freq: dict = defaultdict(int)
        for step, p, row in seq:
            if step < fit_steps and p in ("fwd", "decode"):
                freq[row] += 1
        resident = {r for r, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]}
        for step, p, row in seq:
            score(step, p, row in resident)
    elif policy == "belady":
        nxt = [0] * len(seq)                      # index of each touch's next use of the same row
        last: dict = {}
        for i in range(len(seq) - 1, -1, -1):
            row = seq[i][2]
            nxt[i] = last.get(row, len(seq) + i)
            last[row] = i
        resident: dict = {}                       # row -> its next use
        heap: list = []                           # (-next use, row), lazily invalidated
        for i, (step, p, row) in enumerate(seq):
            hit = row in resident
            score(step, p, hit)
            if not hit and len(resident) >= budget:
                while heap:
                    nu, victim = heapq.heappop(heap)
                    if resident.get(victim) == -nu:
                        del resident[victim]
                        break
            resident[row] = nxt[i]
            heapq.heappush(heap, (-nxt[i], row))
    elif policy == "lru":
        cache: OrderedDict = OrderedDict()
        for step, p, row in seq:
            hit = row in cache
            score(step, p, hit)
            if hit:
                cache.move_to_end(row)
            else:
                if len(cache) >= budget:
                    cache.popitem(last=False)
                cache[row] = True
    elif policy == "lfu":
        count: dict = defaultdict(int)            # frequency over the whole history (ColdTier keeps counts)
        stamp: dict = {}                          # last use, the tie-break
        heap = []                                 # (count, last use, row), lazily invalidated
        resident_set: set = set()
        for t, (step, p, row) in enumerate(seq):
            hit = row in resident_set
            score(step, p, hit)
            count[row] += 1
            stamp[row] = t
            if not hit:
                if len(resident_set) >= budget:
                    while heap:
                        c, s, victim = heapq.heappop(heap)
                        if victim in resident_set and count[victim] == c and stamp[victim] == s:
                            resident_set.discard(victim)
                            break
                resident_set.add(row)
            heapq.heappush(heap, (count[row], t, row))
    else:
        raise ValueError(f"unknown policy {policy!r}")
    out = {p: (hits[p], touches[p]) for p in touches}
    out["pooled"] = (sum(hits.values()), sum(touches.values()))
    return out


def census(seq) -> dict:
    """Distinct experts per (step, layer) in the forward (or decode) pass, and each layer's step-to-step Jaccard."""
    sets: dict = defaultdict(set)
    for step, p, (layer, e) in seq:
        if p in ("fwd", "decode"):
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
    scored_steps = len({s for s, _p, _r in seq if s >= fit_steps})
    out = {"census": census(seq), "row_bytes": row_bytes, "layout": LAYOUT, "fit_steps": fit_steps,
           "scored_steps": scored_steps, "cells": []}
    for b in budgets:
        for pol in policies:
            res = simulate(seq, b, pol, fit_steps)
            per_pass = {p: {"hits": h, "touches": t, "hit_rate": h / t if t else None,
                            "staged_bytes_per_step": (t - h) * row_bytes / scored_steps if scored_steps else None}
                        for p, (h, t) in res.items()}
            out["cells"].append({"budget": b, "policy": pol, "passes": per_pass})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="a training trace .npz from run_capture.py")
    src.add_argument("--decode", help="a #1469 decode trace .jsonl")
    ap.add_argument("--expect-mb", type=int, help="micro-batches per step; required with --trace (the guard)")
    ap.add_argument("--complete-steps", type=int, default=None,
                    help="score only the first N optimizer steps of a capture that ended early (re-checked; see "
                         "load_train_trace_info)")
    ap.add_argument("--truncation-reason", default=None, help="why the capture is cut; required with --complete-steps")
    ap.add_argument("--fit-steps", type=int, default=5)
    ap.add_argument("--null-seed", type=int, default=None, help="replay the shuffled-routing null with this seed")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--moe-intermediate", type=int, default=768)
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.trace:
        if a.expect_mb is None:
            ap.error("--expect-mb is required with --trace")
        mbs, info = load_train_trace_info(a.trace, a.expect_mb, a.complete_steps, a.truncation_reason)
        seq = train_sequence(mbs, null_seed=a.null_seed, n_experts=a.n_experts)
    else:
        if a.complete_steps is not None:
            ap.error("--complete-steps applies to a training trace only")
        seq, info = decode_sequence(a.decode), None
    res = replay(seq, a.fit_steps, expert_row_bytes(a.hidden, a.moe_intermediate))
    res["trace_info"] = info
    res["source"] = a.trace or a.decode
    res["null_seed"] = a.null_seed
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({f"{c['policy']}@{c['budget']}": {p: round(v["hit_rate"], 4) for p, v in c["passes"].items()
                                                       if v["hit_rate"] is not None} for c in res["cells"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
