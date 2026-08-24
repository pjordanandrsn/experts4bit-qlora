# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-b1: routing locality from a --series-out trace (B=1, H arm).

Per layer, over consecutive decode steps: Jaccard overlap of the routed
expert set J(S_t, S_t+1) and the repeat probability P(e in S_t+1 | e in
S_t). This is the evidence bar any future prefetch/prediction line must
clear BEFORE being registered — no speculative prefetch on vibes.

Reads the gzip JSON {"per_layer_series": [[ids...] per step] per layer}.
"""

import argparse
import gzip
import json
import statistics


def locality(series):
    per_layer = []
    for layer in series:
        js, rs = [], []
        for a, b in zip(layer, layer[1:]):
            sa, sb = set(a), set(b)
            if not sa and not sb:
                continue
            inter = len(sa & sb)
            js.append(inter / len(sa | sb))
            if sa:
                rs.append(inter / len(sa))
        per_layer.append({
            "steps": len(layer),
            "jaccard_median": statistics.median(js) if js else None,
            "repeat_prob_median": statistics.median(rs) if rs else None,
            "uniq_median": statistics.median(len(x) for x in layer),
        })
    j_all = [x["jaccard_median"] for x in per_layer
             if x["jaccard_median"] is not None]
    r_all = [x["repeat_prob_median"] for x in per_layer
             if x["repeat_prob_median"] is not None]
    return {
        "layers": len(per_layer),
        "jaccard_median_across_layers":
            statistics.median(j_all) if j_all else None,
        "repeat_prob_median_across_layers":
            statistics.median(r_all) if r_all else None,
        "per_layer": per_layer,
    }


def self_test():
    # identical routing every step -> J=1, repeat=1
    s = [[[1, 2, 3]] * 5]
    out = locality(s)
    assert out["jaccard_median_across_layers"] == 1.0
    assert out["repeat_prob_median_across_layers"] == 1.0
    # disjoint routing every step -> J=0, repeat=0
    s = [[[1, 2], [3, 4], [5, 6], [7, 8]]]
    out = locality(s)
    assert out["jaccard_median_across_layers"] == 0.0
    assert out["repeat_prob_median_across_layers"] == 0.0
    # half overlap: {1,2}->{2,3}: J=1/3, repeat=1/2
    s = [[[1, 2], [2, 3], [3, 4]]]
    out = locality(s)
    assert abs(out["jaccard_median_across_layers"] - 1 / 3) < 1e-9
    assert abs(out["repeat_prob_median_across_layers"] - 0.5) < 1e-9
    print("self-test OK: 3/3")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--series")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.self_test:
        self_test()
        raise SystemExit(0)
    with gzip.open(a.series, "rt") as f:
        data = json.load(f)
    out = locality(data["per_layer_series"])
    text = json.dumps(out, indent=1)
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(text)
    summary = {k: v for k, v in out.items() if k != "per_layer"}
    print(json.dumps(summary, indent=1))
