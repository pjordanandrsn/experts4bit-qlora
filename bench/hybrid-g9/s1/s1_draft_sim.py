# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-s1-acceptance Stage A simulator: prompt-lookup drafting
replayed against recorded greedy traces.

Greedy target + exact-prefix scoring IS speculative acceptance for a
greedy pipeline: the executor would emit the accepted prefix plus the
target's own next token, so tokens/step = m + 1 with m the longest
exact prefix match. No distribution math enters at temperature 0."""

import argparse
import json
import sys
from collections import Counter


def propose(ctx, k, n_min, n_max=3):
    """The registered drafter: longest suffix n-gram (n_max down to
    n_min) with an earlier occurrence; most recent match wins; the k
    tokens FOLLOWING that occurrence are the draft. Empty draft when
    nothing matches."""
    t = len(ctx)
    for n in range(min(n_max, t), n_min - 1, -1):
        suf = tuple(ctx[t - n:t])
        # most recent earlier occurrence: scan right-to-left, the
        # match may not extend past t (the draft source must be
        # strictly historical)
        for i in range(t - n - 1, -1, -1):
            if tuple(ctx[i:i + n]) == suf:
                src = i + n
                return list(ctx[src:src + k])
        # no occurrence at this n: try shorter
    return []


def degeneracy(trace, min_distinct=30, gram=8, max_rep=6):
    """The check-traces law: a repetition-looping or low-entropy trace
    would hand the drafter free matches and fabricate acceptance."""
    if len(set(trace)) < min_distinct:
        return f"only {len(set(trace))} distinct tokens"
    grams = Counter(tuple(trace[i:i + gram])
                    for i in range(len(trace) - gram + 1))
    worst, n = grams.most_common(1)[0] if grams else ((), 0)
    if n > max_rep:
        return f"an {gram}-gram repeats {n}x"
    return None


def simulate(prompt, trace, k, n_min):
    """Walk the trace; at each step draft against everything visible
    (prompt + emitted so far) and score the exact prefix match against
    what the target actually emitted next."""
    ms = []
    for t in range(len(trace)):
        ctx = list(prompt) + trace[:t]
        draft = propose(ctx, k, n_min)
        m = 0
        for d, actual in zip(draft, trace[t:t + len(draft)]):
            if d != actual:
                break
            m += 1
        # the accepted prefix may not run past the trace's end --
        # unverifiable tail is not acceptance
        m = min(m, len(trace) - t)
        ms.append(m)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces_json", nargs="?",
                    help="JSON: [{'prompt': [...], 'tokens': [...]}]")
    ap.add_argument("--out", default="s1_alpha.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.traces_json:
        sys.exit("need traces (or --self-test)")
    traces = json.loads(open(a.traces_json).read())
    kept, dropped = [], []
    for i, tr in enumerate(traces):
        why = degeneracy(tr["tokens"])
        (dropped if why else kept).append((i, tr, why) if why else tr)
    rep = {"n_traces_in": len(traces), "n_traces_kept": len(kept),
           "dropped": [{"idx": i, "why": w} for i, _t, w in dropped],
           "cells": {}}
    if len(kept) < 24:
        rep["refuse"] = (f"only {len(kept)} non-degenerate traces "
                         "(< 24) -- the acceptance table would be "
                         "built on too little text")
    for n_min in (1, 2, 3):
        for k in (4, 8, 16):
            ms = []
            for tr in kept:
                ms.extend(simulate(tr["prompt"], tr["tokens"], k, n_min))
            cell = {"windows": len(ms),
                    "mean_m": (sum(ms) / len(ms)) if ms else None,
                    "tokens_per_step": (sum(ms) / len(ms) + 1.0)
                    if ms else None,
                    "hist": dict(Counter(ms))}
            if len(ms) < 3000:
                cell["refuse"] = f"{len(ms)} windows < 3000"
            rep["cells"][f"n{n_min}_k{k}"] = cell
    open(a.out, "w").write(json.dumps(rep, indent=1))
    print(f"S1SIM traces={rep['n_traces_kept']}/{rep['n_traces_in']}"
          + (" REFUSE" if "refuse" in rep else ""))
    for name, c in rep["cells"].items():
        tps = c["tokens_per_step"]
        print(f"  {name}: windows={c['windows']} "
              f"tokens/step={tps:.3f}" if tps else f"  {name}: no data")


def self_test():
    # drafter: exact repeat means the continuation is predictable
    ctx = [1, 2, 3, 9, 9, 1, 2, 3]
    assert propose(ctx, 4, 1) == [9, 9, 1, 2], propose(ctx, 4, 1)
    # most recent match wins over an older one
    ctx2 = [5, 6, 7, 0, 5, 6, 8, 0, 5, 6]
    assert propose(ctx2, 2, 2) == [8, 0], propose(ctx2, 2, 2)
    # no match at n_min=3 -> empty draft (n=2 would match, must not)
    assert propose([1, 2, 8, 8, 1, 2], 4, 3) == []
    # draft source is strictly historical: suffix matching itself is
    # not a match
    assert propose([4, 5, 6], 4, 1) == []
    # acceptance: at step 0 the context [1,2,3,4] has no repeat yet ->
    # empty draft, m=0; from step 1 the period is visible and windows
    # accept fully; min() caps at the trace end so final windows shrink
    prompt = [1, 2, 3, 4]
    trace = [1, 2, 3, 4, 1, 2, 3, 4]
    ms = simulate(prompt, trace, 4, 1)
    assert ms == [0, 4, 4, 4, 4, 3, 2, 1], ms
    # a draft that diverges at position 0 scores 0
    ms2 = simulate([7, 8, 7, 8], [9, 9, 9, 1, 5, 3, 2, 6], 4, 2)
    assert ms2[0] == 0, ms2
    # degeneracy: a 2-token loop is caught, healthy text is not
    assert degeneracy([1, 2] * 64) is not None
    healthy = list(range(100)) + [5, 17, 3] * 2 + list(range(100, 122))
    assert degeneracy(healthy) is None, degeneracy(healthy)
    # low-distinct refusal fires independently of looping
    assert degeneracy(list(range(10)) * 13) is not None
    print("self-test PASS (drafter recency/floor/history, acceptance "
          "scoring incl. tail cap, both degeneracy modes)")


if __name__ == "__main__":
    main()
