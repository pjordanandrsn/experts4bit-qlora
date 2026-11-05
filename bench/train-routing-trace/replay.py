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

Per-layer budgets (--per-layer-k, decode traces only): K slots in EACH layer, one tier per layer, under three shipped
rules, each class citing the code it re-states:
  lru_seeded   SGLang paged experts: slots seeded with experts 0..K-1, LRU, its wave rule for a step larger than K;
  lru          the vLLM expert pager's VRAM tier: starts empty, LRU;
  lfu_protect  e4b's ColdTier: LFU, an LRU tie-break, the current request's resolved rows never evicted.
Reported per layer and in total, on the same warm scored window: hits, misses and page-ins, with per-token rates.
A #1469 trace stores each routed set sorted, so entries arrive in ascending expert id, not router order. That only
breaks slot ties for the two LRU rules, but lfu_protect's page-ins above misses depend on it: they are an
ascending-order figure, and anything quoting them must say so.

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
    python replay.py --decode qwen3_prose.jsonl --fit-steps 128 --per-layer-k 8,32,64 --out per_layer.json
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


# -- per-layer budgets: three shipped residency rules, K slots in EACH layer -------------------------------------
# One tier per layer, keyed by expert id. Each class re-states one implementation, cited at the commit read:
#   SGLang v0.5.23 (297e54f), PE/ = python/sglang/srt/layers/moe/paged_experts/;
#   vllm-expert-pager e9264fda (tag 0.1.1), src/vllm_expert_pager/;
#   grouped-nf4-gemm 6928361, kernel/nvme_residency.py (e4b's ColdTier).
# A request is one layer's routed entries for one decode step. Each returns (hits, misses, page_ins):
#   hits      routed experts resident when the request starts; misses = distinct routed - hits;
#   page_ins  rows loaded while serving it. It can exceed misses: a rule may evict a row the same request still needs
#             before reaching it (lfu_protect protects only the rows the request has already resolved).
class RequestRefused(ValueError):
    """A request the modelled implementation refuses, or serves on a path this replay does not model."""


class SeededLRU:
    """`lru_seeded`, SGLang paged experts: K slots, seeded with experts 0..K-1 (PE/residency.py:23-30, 62-71), LRU.

    A request whose routed ENTRIES fit the K slots is decided as the device kernel decides it: the default path with
    CUDA graphs on (PE/method.py:177-190, 204-212, 278; kernels/jit/csrc/moe/paged_experts.cuh:26-89). A step clock
    advances; every resident expert of the step is stamped with it; then, in entry order, each missing expert takes the
    least recently used slot not stamped this step, the lowest slot on ties.

    A larger request is planned on the host (PE/executor.py:46-66): the device state loads as an LRU order sorted by
    (last use, slot) (PE/residency.py:77-82); the distinct experts (sorted, torch.unique) split into waves of at most K
    with the residents first (residency.py:49-59); each wave is placed by LRUPolicy.place, its residents refreshed
    before its misses evict (residency.py:32-46); the order is stored back as the newest last uses
    (residency.py:84-99)."""

    def __init__(self, k: int):
        self.k = k
        self.slot_expert = list(range(k))
        self.expert_slot = {e: e for e in range(k)}
        self.lastuse = [0] * k
        self.step = 0

    def request(self, entries):
        distinct = sorted(set(entries))
        hits = sum(e in self.expert_slot for e in distinct)
        loads = self._device(entries) if len(entries) <= self.k else self._waves(distinct)
        return hits, len(distinct) - hits, loads

    def _device(self, entries) -> int:
        self.step += 1
        now = self.step
        for e in entries:
            if e in self.expert_slot:
                self.lastuse[self.expert_slot[e]] = now
        loads = 0
        for e in entries:
            if e in self.expert_slot:
                continue
            free = [(self.lastuse[s], s) for s in range(self.k) if self.lastuse[s] != now]
            if not free:                          # unreachable while entries <= K (paged_experts.cuh:53-54)
                continue
            victim = min(free)[1]
            del self.expert_slot[self.slot_expert[victim]]
            self.slot_expert[victim] = e
            self.expert_slot[e] = victim
            self.lastuse[victim] = now
            loads += 1
        return loads

    def _waves(self, distinct) -> int:
        order = sorted(range(self.k), key=lambda s: (self.lastuse[s], s))
        resident = OrderedDict((self.slot_expert[s], s) for s in order)
        ordered = [e for e in distinct if e in resident] + [e for e in distinct if e not in resident]
        loads = 0
        for wave in [ordered[i:i + self.k] for i in range(0, len(ordered), self.k)]:
            for e in wave:
                if e in resident:
                    resident.move_to_end(e)
            for e in wave:
                if e not in resident:
                    _, slot = resident.popitem(last=False)
                    resident[e] = slot
                    loads += 1
        base = self.step
        for rank, (e, s) in enumerate(resident.items(), start=1):
            self.slot_expert[s] = e
            self.lastuse[s] = base + rank
        self.expert_slot = {e: s for s, e in enumerate(self.slot_expert)}
        self.step = base + self.k
        return loads


class PagerLRU:
    """`lru`, the vLLM expert pager's VRAM tier: one ExpertTable per layer (table.py:19-128; the decode kernel matches
    it, acquire.py:3-6, 25-99). It starts empty (table.py:36-44). Hits keep their slots. The misses, in ascending expert
    id (the cumsum rank, table.py:109), take the slots in ascending (last use, slot) order with every slot holding a hit
    pushed to the end (table.py:99-110); every needed slot is stamped with the step (table.py:118-120). A request with
    more routed entries than K is not the decode path (acquire.py:3, 172-173) and is refused."""

    def __init__(self, k: int):
        self.k = k
        self.slot_of: dict = {}
        self.expert_in = [-1] * k
        self.last_used = [0] * k
        self.step = 1

    def request(self, entries):
        if len(entries) > self.k:
            raise RequestRefused(f"{len(entries)} routed entries > K={self.k}: the pager's prefill path, not modelled")
        need = sorted(set(entries))
        hit = [e for e in need if e in self.slot_of]
        miss = [e for e in need if e not in self.slot_of]
        used = {self.slot_of[e] for e in hit}
        order = sorted(range(self.k), key=lambda s: (s in used, self.last_used[s], s))
        for e, victim in zip(miss, order):
            old = self.expert_in[victim]
            if old >= 0:
                del self.slot_of[old]
            self.slot_of[e] = victim
            self.expert_in[victim] = e
        for e in need:
            self.last_used[self.slot_of[e]] = self.step
        self.step += 1
        return len(hit), len(miss), len(miss)


class ColdTierLFU:
    """`lfu_protect`, e4b's ColdTier (nvme_residency.py:23-31): LFU, an LRU tie-break, the current request protected.

    Per request (one demand `ensure`, no prefetcher): a request clock; every routed pick adds to its expert's
    frequency, kept over the whole history, and is stamped with the clock BEFORE any slot is chosen (:452-456); the
    previous request's window is released (:457-460). The unique experts resolve in request order (:436-437,
    :464-475). A miss takes a free slot, the free list popping the highest slot first (:147, :654-655), else the
    victim: the lowest (frequency, last use, slot) among the slots the request has not resolved yet (:294-328, its
    heap twin :330-391, called at :657). So protection covers the rows the request has resolved so far, and a resident
    row it has not reached yet is a candidate. A request with more unique rows than K is refused (:438-441)."""

    def __init__(self, k: int):
        self.k = k
        self.key_of: list = [None] * k
        self.slot_of: dict = {}
        self.free = list(range(k))
        self.freq: dict = defaultdict(int)
        self.last: dict = {}
        self.clock = 0

    def request(self, entries):
        uniq = list(dict.fromkeys(entries))
        if len(uniq) > self.k:
            raise RequestRefused(f"{len(uniq)} unique rows > K={self.k}: ColdTier refuses the request")
        hits = sum(e in self.slot_of for e in uniq)
        self.clock += 1
        for e in entries:
            self.freq[e] += 1
            self.last[e] = self.clock
        own: set = set()
        loads = 0
        for e in uniq:
            slot = self.slot_of.get(e)
            if slot is None:
                slot = self.free.pop() if self.free else self._victim(own)
                old = self.key_of[slot]
                if old is not None:
                    del self.slot_of[old]
                self.key_of[slot] = e
                self.slot_of[e] = slot
                loads += 1
            own.add(slot)
        return hits, len(uniq) - hits, loads

    def _victim(self, own: set) -> int:
        cands = [(self.freq[e], self.last[e], s) for s, e in enumerate(self.key_of) if s not in own]
        return min(cands)[2]


PER_LAYER_POLICIES = {"lru_seeded": SeededLRU, "lru": PagerLRU, "lfu_protect": ColdTierLFU}


def decode_requests(path: str):
    """(meta, [(step, layer, routed experts)]) from a #1469 decode trace, in execution order: per decoded token, layers
    0 .. L-1. The trace stores each layer's routed set sorted, so the router's score order is not recoverable: entry
    order is ascending expert id. For lru_seeded and lru the order only decides slot ties. For lfu_protect it also
    decides which resident rows a miss may evict before the request reaches them, so its page-ins above misses are an
    ascending-order figure, not what ColdTier sees under router order."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    meta = rows[0].get("meta", {})
    reqs = [(int(r["step"]), int(layer), sorted(int(e) for e in r["routed"][layer]))
            for r in rows[1:] for layer in sorted(r["routed"], key=int)]
    return meta, reqs


def _tally(c: list, tokens: int) -> dict:
    requests, routed, hits, misses, page_ins = c    # routed = DISTINCT routed experts (a B=1 request is a set)
    return {"requests": requests, "routed": routed, "hits": hits, "misses": misses, "page_ins": page_ins,
            "hit_rate": hits / routed if routed else None,
            "per_token": {"hits": hits / tokens, "misses": misses / tokens, "page_ins": page_ins / tokens}
            if tokens else None}


def simulate_per_layer(requests, k: int, policy: str, fit_steps: int) -> dict:
    """One `policy` tier of K slots per layer, run over every request (warm through the fit window) and tallied on
    steps >= fit_steps: per layer and in total, with per-token rates (a token = one decode step)."""
    make = PER_LAYER_POLICIES[policy]
    tiers: dict = {}
    counts: dict = defaultdict(lambda: [0, 0, 0, 0, 0])
    tokens: set = set()
    for step, layer, entries in requests:
        if layer not in tiers:
            tiers[layer] = make(k)
        hits, misses, page_ins = tiers[layer].request(entries)
        if step >= fit_steps:
            tokens.add(step)
            c = counts[layer]
            c[0] += 1
            c[1] += hits + misses
            c[2] += hits
            c[3] += misses
            c[4] += page_ins
    total = [sum(c[i] for c in counts.values()) for i in range(5)]
    return {"tokens": len(tokens), "total": _tally(total, len(tokens)),
            "layers": {layer: _tally(counts[layer], len(tokens)) for layer in sorted(counts)}}


def replay_per_layer(requests, ks, fit_steps: int, row_bytes: int, top_k: int,
                     policies=tuple(PER_LAYER_POLICIES)) -> dict:
    """Every policy at every per-layer K. A K below the router's top_k is refused up front: SGLang refuses it at load
    (PE/method.py:336-339), and the pager and ColdTier refuse a B=1 decode request larger than K."""
    low = [k for k in ks if k < top_k]
    if low:
        raise RequestRefused(f"per-layer K {low} < top_k={top_k}")
    out = {"row_bytes": row_bytes, "layout": LAYOUT, "fit_steps": fit_steps, "top_k": top_k, "cells": []}
    for k in ks:
        for pol in policies:
            res = simulate_per_layer(requests, k, pol, fit_steps)
            per_tok = res["total"]["per_token"]
            res.update({"k": k, "policy": pol,
                        "page_in_bytes_per_token": per_tok["page_ins"] * row_bytes if per_tok else None})
            out["cells"].append(res)
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
    ap.add_argument("--per-layer-k", default=None,
                    help="comma-separated K slots PER LAYER: replay a decode trace through lru_seeded, lru and "
                         "lfu_protect instead of the global budgets")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.per_layer_k is not None:
        if not a.decode:
            ap.error("--per-layer-k applies to a decode trace only")
        meta, reqs = decode_requests(a.decode)
        if "top_k" not in meta:
            ap.error("--per-layer-k needs the trace meta's top_k")
        ks = [int(k) for k in a.per_layer_k.split(",")]
        res = replay_per_layer(reqs, ks, a.fit_steps, expert_row_bytes(a.hidden, a.moe_intermediate),
                               int(meta["top_k"]))
        res["source"] = a.decode
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({f"{c['policy']}@K{c['k']}": round(c["total"]["hit_rate"], 4) for c in res["cells"]
                          if c["total"]["hit_rate"] is not None}))
        return 0
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
