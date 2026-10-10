"""The pre-registered cross-step residency bar, computed from a training routing trace. CPU only, no timings.

The bar (fixed before capture): at a fixed row budget, run a policy over every pass of the trace through the end of
step s, FREEZE its resident set, and score step s+1's FORWARD touches against it. A touch is one distinct
(layer, expert) row per micro-batch, the replay's unit; all micro-batches of step s+1 are scored. Within-step reuse
(the recompute and dgrad of a micro-batch, and later micro-batches of the same step) is excluded by construction:
the cache never sees step s+1 before it is scored.

    BUILD-SIGNAL  iff  LFU hit rate >= 2.0 x the shuffled null's  AND  median per-layer step-to-step Jaccard >= 0.5
    NO-BUILD      iff  LFU hit rate <= 1.25 x null, or Belady (the ceiling) <= 1.25 x null
    INCONCLUSIVE  otherwise

The sequence and the null come from replay.py unchanged (`train_sequence`); the policies are re-stated here only so
their state can be frozen at step boundaries, and the tests check they make the same hit/miss
decisions as replay.simulate on the live sequence.

    python cross_step_bar.py --trace trace.npz --expect-mb 8 --budget 768 --first-scored 5 --null-seed 1 --out bar.json
    (a capture that ended early: add replay.py's --complete-steps N --truncation-reason TEXT; same checks, recorded)
"""
from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _replay():
    spec = importlib.util.spec_from_file_location("replay", os.path.join(HERE, "replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _LFU:
    """LFU with LRU tie-break over the whole history: the same rule as replay.simulate's "lfu"."""

    def __init__(self, budget):
        self.budget, self.count, self.stamp, self.heap, self.resident, self.t = budget, defaultdict(int), {}, [], set(), 0

    def touch(self, row):
        hit = row in self.resident
        self.count[row] += 1
        self.stamp[row] = self.t
        if not hit:
            if len(self.resident) >= self.budget:
                while self.heap:
                    c, s, victim = heapq.heappop(self.heap)
                    if victim in self.resident and self.count[victim] == c and self.stamp[victim] == s:
                        self.resident.discard(victim)
                        break
            self.resident.add(row)
        heapq.heappush(self.heap, (self.count[row], self.t, row))
        self.t += 1
        return hit


class _LRU:
    def __init__(self, budget):
        self.budget, self.cache = budget, OrderedDict()

    @property
    def resident(self):
        return set(self.cache)

    def touch(self, row):
        hit = row in self.cache
        if hit:
            self.cache.move_to_end(row)
        else:
            if len(self.cache) >= self.budget:
                self.cache.popitem(last=False)
            self.cache[row] = True
        return hit


class _Belady:
    """Evict the row whose next use is farthest. Needs the whole sequence up front (the offline ceiling)."""

    def __init__(self, budget, seq):
        self.budget, self.resident_map, self.heap, self.i = budget, {}, [], 0
        self.nxt = [0] * len(seq)
        last: dict = {}
        for i in range(len(seq) - 1, -1, -1):
            row = seq[i][2]
            self.nxt[i] = last.get(row, len(seq) + i)
            last[row] = i

    @property
    def resident(self):
        return set(self.resident_map)

    def touch(self, row):
        hit = row in self.resident_map
        if not hit and len(self.resident_map) >= self.budget:
            while self.heap:
                nu, victim = heapq.heappop(self.heap)
                if self.resident_map.get(victim) == -nu:
                    del self.resident_map[victim]
                    break
        self.resident_map[row] = self.nxt[self.i]
        heapq.heappush(self.heap, (-self.nxt[self.i], row))
        self.i += 1
        return hit


def make_policy(name, budget, seq):
    return {"lfu": lambda: _LFU(budget), "lru": lambda: _LRU(budget), "belady": lambda: _Belady(budget, seq)}[name]()


def cross_step(seq, budget: int, policy: str, first_scored: int) -> dict:
    """Hit rate of step s+1's forward touches against the resident set frozen at the end of step s, for every
    s+1 >= first_scored. The policy itself runs over every pass (fwd, recompute, dgrad), as the step does.

    Belady's resident set at the end of step s is chosen with knowledge of the future, including step s+1: that is
    what makes it the ceiling, not a deployable policy."""
    pol = make_policy(policy, budget, seq)
    by_step: dict = defaultdict(list)
    for i, (step, p, row) in enumerate(seq):
        by_step[step].append((i, p, row))
    steps = sorted(by_step)
    hits = touches = 0
    per_step = {}
    for s in steps:
        if s >= first_scored and s - 1 in by_step:
            frozen = pol.resident
            fwd = [row for _i, p, row in by_step[s] if p == "fwd"]
            h = sum(row in frozen for row in fwd)
            per_step[s] = h / len(fwd) if fwd else None
            hits += h
            touches += len(fwd)
        for _i, _p, row in by_step[s]:
            pol.touch(row)
    return {"hits": hits, "touches": touches, "hit_rate": hits / touches if touches else None, "per_step": per_step}


def oracle_frozen(seq, budget: int, first_scored: int) -> float | None:
    """REPORTING: the ceiling of the frozen metric. For each scored step, the best possible frozen set of `budget` rows
    is that step's own most-touched forward rows; no policy's frozen set can score more. (Belady is optimal for the
    LIVE cache over the whole sequence, not for this frozen score, so it is not this metric's ceiling.)"""
    fwd: dict = defaultdict(lambda: defaultdict(int))
    for step, p, row in seq:
        if p == "fwd":
            fwd[step][row] += 1
    hits = touches = 0
    for s in sorted(fwd):
        if s < first_scored:
            continue
        counts = sorted(fwd[s].values(), reverse=True)
        hits += sum(counts[:budget])
        touches += sum(counts)
    return hits / touches if touches else None


def median_jaccard(seq) -> float | None:
    """Median over (layer, s -> s+1) of the Jaccard of the per-step forward sets (replay.census's sets)."""
    sets: dict = defaultdict(set)
    for step, p, (layer, e) in seq:
        if p == "fwd":
            sets[(step, layer)].add(e)
    steps = sorted({s for s, _ in sets})
    layers = sorted({layer for _, layer in sets})
    jac = [len(sets[(a, la)] & sets[(b, la)]) / len(sets[(a, la)] | sets[(b, la)])
           for a, b in zip(steps, steps[1:]) for la in layers if sets[(a, la)] | sets[(b, la)]]
    return float(np.median(jac)) if jac else None


def distinct_per_mb_layer(mbs) -> dict:
    """How many of the experts each (micro-batch, layer) forward touches: the saturation read, reported first."""
    d = [len(np.unique(ids)) for v in mbs.values() for ids in v["fwd"].values()]
    return {"mean": float(np.mean(d)), "min": int(min(d)), "max": int(max(d)), "n": len(d)}


def skew(mbs, n_experts: int, top: int = 16) -> dict:
    """REPORTING only (compute and load balance, not transfer: staging cost depends on WHICH experts are touched, not
    on how many tokens each gets). Per (step, layer): the share of the forward's routed tokens on its `top` busiest
    experts; and per layer, the Jaccard of consecutive steps' top-`top` sets (their stability)."""
    counts: dict = defaultdict(lambda: np.zeros(n_experts, np.int64))
    for (step, _mb), v in mbs.items():
        for layer, ids in v["fwd"].items():
            counts[(step, layer)] += np.bincount(np.asarray(ids, np.int64).ravel(), minlength=n_experts)
    share, tops = [], {}
    for k, c in counts.items():
        order = np.argsort(-c, kind="stable")[:top]
        share.append(c[order].sum() / c.sum())
        tops[k] = set(order.tolist())
    steps = sorted({s for s, _ in tops})
    layers = sorted({la for _, la in tops})
    stab = [len(tops[(a, la)] & tops[(b, la)]) / len(tops[(a, la)] | tops[(b, la)])
            for a, b in zip(steps, steps[1:]) for la in layers]

    def q(xs):
        return {"median": float(np.median(xs)), "p10": float(np.percentile(xs, 10)),
                "p90": float(np.percentile(xs, 90))} if xs else None
    return {"top": top, "uniform_share": top / n_experts, "top_share": q(share), "top_set_jaccard": q(stab)}


def verdict(lfu: float, null: float, belady: float, jac: float) -> str:
    if belady <= 1.25 * null or lfu <= 1.25 * null:
        return "NO-BUILD"
    if lfu >= 2.0 * null and jac >= 0.5:
        return "BUILD-SIGNAL"
    return "INCONCLUSIVE"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--expect-mb", type=int, required=True)
    ap.add_argument("--budget", type=int, default=768)
    ap.add_argument("--first-scored", type=int, default=5, help="0-based index of the first scored step (6th step)")
    ap.add_argument("--null-seed", type=int, default=1)
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--complete-steps", type=int, default=None, help="replay.py's cut: score the first N complete steps")
    ap.add_argument("--truncation-reason", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rp = _replay()
    mbs, info = rp.load_train_trace_info(a.trace, a.expect_mb, a.complete_steps, a.truncation_reason)
    real = rp.train_sequence(mbs, n_experts=a.n_experts)
    null = rp.train_sequence(mbs, null_seed=a.null_seed, n_experts=a.n_experts)
    res = {"trace_info": info, "budget": a.budget, "first_scored_index": a.first_scored, "null_seed": a.null_seed,
           "saturation": distinct_per_mb_layer(mbs),
           "median_jaccard": median_jaccard(real), "median_jaccard_null": median_jaccard(null),
           "skew_reporting": skew(mbs, a.n_experts)}
    for name, seq in (("real", real), ("null", null)):
        res[name] = {pol: cross_step(seq, a.budget, pol, a.first_scored) for pol in ("lfu", "lru", "belady")}
        res[name]["oracle_frozen_reporting"] = oracle_frozen(seq, a.budget, a.first_scored)
    hr = {k: res[k]["lfu"]["hit_rate"] for k in ("real", "null")}
    res["verdict"] = verdict(hr["real"], hr["null"], res["real"]["belady"]["hit_rate"], res["median_jaccard"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("saturation", "median_jaccard", "verdict", "skew_reporting")}),
          {k: {p: res[k][p]["hit_rate"] for p in ("lfu", "lru", "belady")} for k in ("real", "null")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
