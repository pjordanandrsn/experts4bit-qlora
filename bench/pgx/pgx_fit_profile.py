"""PGX: E-hyb's routing profile, written at $0 from the #1469 Qwen3-30B-A3B decode traces.

    python bench/pgx/pgx_fit_profile.py --out bench/pgx/fit_profile.jsonl TRACE.jsonl...

Each trace is a grouped-nf4-gemm ``bench/cold-engine/routing-trace/qwen3_<kind>.jsonl``: a meta line, then one line per
decode step with ``routed = {layer index: [expert ids]}``. Every routed pick adds 1 to its (layer, expert). The output
is an ``engines/expert_profile`` JSONL, the format ``E4B_PAGED_PROFILE`` hands to the placement solver (e4b#2169):
one ``row: meta`` line naming the sources and their sha256, then ``row: expert`` lines with ``layer_id`` (the MoE
ordinal: for Qwen3-30B-A3B every decoder layer is MoE, so it is the trace's layer index), ``expert_id`` and
``tokens_routed``.

Refuses (exit 2) a trace whose sha256 is not in ``--expect-sha`` when given, a layer or expert outside
``--layers`` x ``--experts``, and any layer with no routed pick: ``serve_paged.placement_profile`` would accept a
partial profile, and the solver would then place that layer by its tie-break.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit(paths, layers: int, experts: int) -> tuple[Counter, dict]:
    counts: Counter = Counter()
    steps = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        rows = [r for r in lines if "routed" in r]
        steps[os.path.basename(path)] = len(rows)
        for r in rows:
            for lay, ids in r["routed"].items():
                for e in ids:
                    key = (int(lay), int(e))
                    if not (0 <= key[0] < layers and 0 <= key[1] < experts):
                        raise ValueError(f"{path}: (layer, expert) {key} outside {layers} x {experts}")
                    counts[key] += 1
    empty = [lay for lay in range(layers) if not any(counts[(lay, e)] for e in range(experts))]
    if empty:
        raise ValueError(f"no routed pick in layer(s) {empty[:5]}{'...' if len(empty) > 5 else ''}: a partial profile")
    return counts, steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--expect-sha", default="", help="comma-separated sha256 prefixes every trace must match one of")
    a = ap.parse_args(argv)
    shas = {os.path.basename(p): sha256(p) for p in a.traces}
    want = [w for w in a.expect_sha.split(",") if w]
    if want:
        bad = [n for n, s in shas.items() if not any(s.startswith(w) for w in want)]
        if bad:
            print(f"REFUSED: trace(s) {bad} not in --expect-sha", file=sys.stderr)
            return 2
    try:
        counts, steps = fit(a.traces, a.layers, a.experts)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    meta = {"row": "meta", "phase": "pgx-fit", "source": "#1469 decode traces", "traces": shas, "steps": steps,
            "layers": a.layers, "experts": a.experts, "picks": sum(counts.values())}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, sort_keys=True) + "\n")
        for (lay, e) in sorted(counts):
            f.write(json.dumps({"row": "expert", "layer_id": lay, "expert_id": e, "tokens_routed": counts[(lay, e)]},
                               sort_keys=True) + "\n")
    print(f"wrote {a.out}: {len(counts)} (layer, expert) rows, {meta['picks']} picks from {len(a.traces)} traces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
