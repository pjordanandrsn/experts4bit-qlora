#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""locality_summary.py -- #1469 item 1: the expert-locality summaries from a raw routing trace (CPU, numpy only).

**Input:** the census's ``.npz``:
- ``expert_ids`` (int16, ``[n_tokens, n_layers, top_k]``, in token order);
- ``seq_offsets`` (int32, ``[n_seqs + 1]``);
- ``n_experts`` (int).

The format was agreed with the owner of #1469 items 2 and 3.

**Per layer, within sequences only** (a window never crosses a sequence boundary):
- **Union.** For W in 1, 16, 32, 64 and 128, the number of distinct experts per W-token window. The windows are
  consecutive and non-overlapping; a sequence's partial tail is dropped. Reported: mean, p50, p95, max and the window
  count.
- **Churn.** The mean fraction of the k routed ids that change from one token to the next: ``|S_t - S_{t-1}| / k``.

**Hit rate.** The residency policy's own counters (decode only), when the manifest carries them:
``1 - cold_gathers / (steps * k)`` per layer. Here ``cold_gathers = cold_pcie_bytes / row_bytes`` from the pipelined
engine's ``traffic()``. It is derived from the counters, never re-simulated.

    locality_summary.py --npz trace.npz [--manifest manifest.json] --phase decode --out summaries.json
    locality_summary.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys

WINDOWS = (1, 16, 32, 64, 128)


def _pct(xs, p):
    import numpy as np
    return float(np.percentile(np.asarray(xs, dtype=float), p, method="linear")) if len(xs) else None


def unions(ids, offsets, layer: int, w: int) -> list:
    """Distinct experts per non-overlapping w-token window of ``layer``, within each sequence."""
    out = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        seq = ids[a:b, layer, :]
        for s in range(0, (b - a) - w + 1, w):
            out.append(len(set(seq[s:s + w].reshape(-1).tolist())))
    return out


def churn(ids, offsets, layer: int) -> float | None:
    """Mean fraction of the k ids that change token to token, within sequences."""
    k = ids.shape[2]
    changes, pairs = 0, 0
    for a, b in zip(offsets[:-1], offsets[1:]):
        for t in range(a + 1, b):
            changes += len(set(ids[t, layer].tolist()) - set(ids[t - 1, layer].tolist()))
            pairs += 1
    return round(changes / (pairs * k), 6) if pairs else None


def summarize(ids, offsets, *, counters: dict | None = None) -> dict:
    """{"layers": n, "top_k": k, "union": {layer: {W: stats}}, "churn": {layer: x}, "hit_rate": {layer: x} | None}."""
    n_tok, n_layers, k = ids.shape
    if int(offsets[0]) != 0 or int(offsets[-1]) != n_tok or any(b < a for a, b in zip(offsets[:-1], offsets[1:])):
        raise ValueError(f"seq_offsets {list(offsets)[:4]}... do not partition {n_tok} tokens")
    rep = {"layers": n_layers, "top_k": k, "tokens": n_tok, "sequences": len(offsets) - 1, "windows": list(WINDOWS),
           "union": {}, "churn": {}, "hit_rate": None}
    for layer in range(n_layers):
        rep["union"][str(layer)] = {}
        for w in WINDOWS:
            u = unions(ids, offsets, layer, w)
            rep["union"][str(layer)][str(w)] = {"windows": len(u), "mean": round(sum(u) / len(u), 4) if u else None,
                                                "p50": _pct(u, 50), "p95": _pct(u, 95), "max": max(u) if u else None}
        rep["churn"][str(layer)] = churn(ids, offsets, layer)
    if counters:
        rep["hit_rate"] = hit_rates(counters, k)
    return rep


def hit_rates(counters: dict, k: int) -> dict:
    """``counters``: {"steps": n, "row_bytes": {layer: b}, "cold_pcie_bytes": {layer: b}} from the pipelined engine."""
    steps, out = int(counters["steps"]), {}
    for layer, cold in counters["cold_pcie_bytes"].items():
        rb = int(counters["row_bytes"][layer])
        if rb <= 0 or cold % rb:
            raise ValueError(f"layer {layer}: cold_pcie_bytes {cold} is not a whole number of {rb}-byte rows")
        gathers = cold // rb
        if gathers > steps * k:
            raise ValueError(f"layer {layer}: {gathers} cold gathers exceed {steps} steps x k={k}")
        out[str(layer)] = {"cold_gathers": gathers, "lanes": steps * k, "hit_rate": round(1 - gathers / (steps * k), 6)}
    return out


def self_test() -> int:
    import numpy as np
    ok = []
    # two sequences, 2 layers, k=2: layer 0 repeats one pair, layer 1 rotates through 4 experts
    s1 = np.array([[[0, 1], [0, 1]], [[0, 1], [2, 3]], [[0, 1], [0, 1]], [[0, 1], [2, 3]]], dtype=np.int16)
    s2 = np.array([[[5, 6], [4, 5]], [[5, 6], [6, 7]]], dtype=np.int16)
    ids, off = np.concatenate([s1, s2]), np.array([0, 4, 6], dtype=np.int32)
    ok.append(unions(ids, off, 0, 1) == [2] * 6 and unions(ids, off, 1, 2) == [4, 4, 4])
    ok.append(unions(ids, off, 1, 4) == [4])                         # the 2-token second sequence has no 4-window
    ok.append(churn(ids, off, 0) == 0.0 and churn(ids, off, 1) == 1.0)   # never across the sequence boundary
    r = summarize(ids, off)
    ok.append(r["union"]["1"]["16"]["windows"] == 0 and r["union"]["1"]["16"]["mean"] is None
              and r["union"]["0"]["1"]["mean"] == 2.0 and r["sequences"] == 2)
    try:
        summarize(ids, np.array([0, 4, 5], dtype=np.int32))
        ok.append(False)
    except ValueError:
        ok.append(True)                                              # offsets that do not cover the trace
    c = {"steps": 10, "row_bytes": {"0": 100, "1": 100}, "cold_pcie_bytes": {"0": 2000, "1": 0}}
    h = hit_rates(c, 4)
    ok.append(h["0"] == {"cold_gathers": 20, "lanes": 40, "hit_rate": 0.5} and h["1"]["hit_rate"] == 1.0)
    for bad in ({"steps": 10, "row_bytes": {"0": 100}, "cold_pcie_bytes": {"0": 150}},     # not whole rows
                {"steps": 1, "row_bytes": {"0": 100}, "cold_pcie_bytes": {"0": 900}}):     # more gathers than lanes
        try:
            hit_rates(bad, 4)
            ok.append(False)
        except ValueError:
            ok.append(True)
    ok.append(summarize(ids, off, counters={"steps": 3, "row_bytes": {"0": 8}, "cold_pcie_bytes": {"0": 16}})
              ["hit_rate"]["0"]["hit_rate"] == round(1 - 2 / 6, 6))
    print(f"locality_summary self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz")
    ap.add_argument("--manifest")
    ap.add_argument("--phase", choices=("decode", "prefill"))
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.npz and a.phase and a.out):
        ap.error("--npz --phase --out, or --self-test")
    import numpy as np
    z = np.load(a.npz)
    counters = None
    if a.manifest and a.phase == "decode":
        counters = json.load(open(a.manifest)).get("residency_counters")
    rep = summarize(z["expert_ids"], z["seq_offsets"], counters=counters)
    rep["phase"] = a.phase
    json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"LOCALITY_SUMMARY {a.phase} layers={rep['layers']} tokens={rep['tokens']} sequences={rep['sequences']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
