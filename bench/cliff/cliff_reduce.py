#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""cliff_reduce.py -- the W1/W2/W4 cliff census's tabulation (bench/cliff/README.md). EXPLORE tier: never a reading.

It reads what the runner leaves (arm_S1a.json, arm_S1b.json, pername_b{1,2,4}.json) and writes:
  steps        each batch's UNPROFILED step (the mean of the S1a / S1b arms' decode_ms_per_step), with the self-pair
               ratio; W16 is carried as context.
  batches      per batch: the per-name table's P_ms, steps kept, class_removable_ms, and every name's removable ms and
               launches per step (summed over grids: a name's grid changes with the batch).
  cliff        per name and per class, removable(W2) - removable(W1) and removable(W4) - removable(W1), largest first,
               with each line's share of the unprofiled step difference. This is the question: where the cliff's time goes.
verdict COMPLETE when both arms and all three tables are present and ok, else INCOMPLETE (with why). Nothing is proposed.

    python cliff_reduce.py --dir /root/cliff --out verdict_cliff.json --e4b-sha <sha>
    python cliff_reduce.py --self-test
"""
import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

BATCHES = (1, 2, 4)
ARMS = ("S1a", "S1b")


def _load(root, name):
    try:
        return json.loads((Path(root) / name).read_text())
    except (OSError, ValueError):
        return None


def steps(root) -> dict:
    arms = {t: _load(root, f"arm_{t}.json") for t in ARMS}
    out = {}
    for w in ("W1", "W2", "W4", "W16"):
        vals = [((a or {}).get("workloads") or {}).get(w, {}).get("decode_ms_per_step") for a in arms.values()]
        if all(isinstance(v, (int, float)) for v in vals):
            out[w] = {"mean_ms": round(sum(vals) / len(vals), 4), "S1a": vals[0], "S1b": vals[1],
                      "self_pair": round(vals[0] / vals[1], 4) if vals[1] else None}
        else:
            out[w] = None
    return out


def by_name(table) -> dict:
    acc = defaultdict(lambda: {"removable_ms": 0.0, "n_per_step": 0.0, "class": None})
    for r in table.get("rows", []):
        k = r["name"] + (f"|{r['role']}" if r.get("role") else "")
        a = acc[k]
        a["removable_ms"] += r.get("removable_ms") or 0.0
        a["n_per_step"] += r.get("n_per_step") or 0.0
        a["class"] = a["class"] or r.get("class")
    return {k: {"removable_ms": round(v["removable_ms"], 6), "n_per_step": v["n_per_step"], "class": v["class"]}
            for k, v in acc.items()}


def diff(lo: dict, hi: dict, field: str, denom) -> list:
    keys = set(lo) | set(hi)
    rows = []
    for k in keys:
        a = (lo.get(k) or {}).get(field, 0.0) if isinstance(lo.get(k), dict) else (lo.get(k) or 0.0)
        b = (hi.get(k) or {}).get(field, 0.0) if isinstance(hi.get(k), dict) else (hi.get(k) or 0.0)
        d = round(b - a, 6)
        rows.append({"key": k, "lo_ms": round(a, 6), "hi_ms": round(b, 6), "delta_ms": d,
                     "share_of_step_delta": round(d / denom, 4) if denom else None})
    rows.sort(key=lambda r: -r["delta_ms"])
    return rows


def reduce(root, e4b_sha=None) -> dict:
    why = []
    st = steps(root)
    for w in ("W1", "W2", "W4"):
        if st.get(w) is None:
            why.append(f"unprofiled {w} missing")
    tables = {}
    for b in BATCHES:
        t = _load(root, f"pername_b{b}.json")
        if not t or t.get("status") != "ok":
            why.append(f"pername_b{b} {'missing' if not t else t.get('status')}: {(t or {}).get('why')}")
        else:
            tables[b] = t
    batches = {}
    for b, t in tables.items():
        batches[f"W{b}"] = {"P_ms": t.get("P_ms"), "steps_kept": t.get("steps_kept"),
                            "kernels_per_graph_modal": t.get("kernels_per_graph_modal"),
                            "class_removable_ms": t.get("class_removable_ms"), "names": by_name(t)}
    cliff = {}
    if 1 in tables:
        for b in (2, 4):
            if b not in tables:
                continue
            lo, hi = batches["W1"], batches[f"W{b}"]
            dstep = (st[f"W{b}"]["mean_ms"] - st["W1"]["mean_ms"]) if st.get("W1") and st.get(f"W{b}") else None
            cliff[f"W{b}-W1"] = {"step_delta_ms": round(dstep, 4) if dstep is not None else None,
                                 "P_delta_ms": round(hi["P_ms"] - lo["P_ms"], 6),
                                 "by_class": diff(lo["class_removable_ms"] or {}, hi["class_removable_ms"] or {}, None, dstep),
                                 "by_name": diff(lo["names"], hi["names"], "removable_ms", dstep)}
    return {"lane": "cliff (EXPLORE, not a reading)", "e4b_sha": e4b_sha,
            "verdict": "COMPLETE" if not why else "INCOMPLETE", "why": why,
            "steps": st, "batches": batches, "cliff": cliff}


def summary(v) -> str:
    out = [f"CLIFF {v['verdict']} {v['why'] or ''}".rstrip()]
    for w, s in v["steps"].items():
        out.append(f"  step {w}: {s['mean_ms']} ms (self-pair {s['self_pair']})" if s else f"  step {w}: missing")
    for k, c in v["cliff"].items():
        out.append(f"  {k}: step +{c['step_delta_ms']} ms, profiled period +{c['P_delta_ms']} ms")
        out.append("    by class: " + ", ".join(f"{r['key']} {r['delta_ms']:+.3f}" for r in c["by_class"][:6]))
        out.append("    by name:  " + ", ".join(f"{r['key']} {r['delta_ms']:+.3f}" for r in c["by_name"][:8]))
    return "\n".join(out)


def self_test() -> int:
    def table(rows, P):
        cls = defaultdict(float)
        for r in rows:
            cls[r["class"]] += r["removable_ms"]
        return {"status": "ok", "steps_kept": 60, "P_ms": P, "kernels_per_graph_modal": len(rows),
                "class_removable_ms": dict(cls), "rows": rows}

    def row(name, grid, cls, ms, n=48, role=None):
        return {"key": f"{name}|{grid}" + (f"|{role}" if role else ""), "name": name, "grid": grid, "role": role,
                "class": cls, "removable_ms": ms, "n_per_step": n}

    cases = []
    with tempfile.TemporaryDirectory() as d:
        D = Path(d)
        for t, k in (("S1a", 1.0), ("S1b", 1.01)):
            (D / f"arm_{t}.json").write_text(json.dumps({"workloads": {
                "W1": {"decode_ms_per_step": 4.0 * k}, "W2": {"decode_ms_per_step": 6.0 * k},
                "W4": {"decode_ms_per_step": 7.0 * k}, "W16": {"decode_ms_per_step": 11.0 * k}}}))
        (D / "pername_b1.json").write_text(json.dumps(table([row("_gemv_nf4_bw", "(1,1)", "moe_expert", 1.0),
                                                             row("cublas_gemv", "(9,1)", "dense_gemm", 1.8, 97)], 3.9)))
        (D / "pername_b2.json").write_text(json.dumps(table([row("_k25", "(2,1)", "moe_expert", 1.6),
                                                             row("cublas_gemm", "(9,2)", "dense_gemm", 2.6, 97),
                                                             row("cublas_gemm", "(9,3)", "dense_gemm", 0.2, 1, "head")], 5.9)))
        v = reduce(d, "f" * 40)
        cases.append(("an incomplete set says INCOMPLETE", v["verdict"] == "INCOMPLETE" and any("pername_b4" in w for w in v["why"])))
        (D / "pername_b4.json").write_text(json.dumps(table([row("_k25", "(4,1)", "moe_expert", 1.9),
                                                             row("cublas_gemm", "(9,4)", "dense_gemm", 2.9, 97)], 6.9)))
        v = reduce(d, "f" * 40)
        c2 = v["cliff"]["W2-W1"]
        cases += [
            ("a full set is COMPLETE", v["verdict"] == "COMPLETE" and not v["why"]),
            ("the step is the arms' mean", abs(v["steps"]["W2"]["mean_ms"] - 6.03) < 1e-9),
            ("the self-pair is S1a over S1b", abs(v["steps"]["W1"]["self_pair"] - round(1 / 1.01, 4)) < 1e-9),
            ("the step delta is W2 minus W1", abs(c2["step_delta_ms"] - 2.01) < 1e-9),
            ("a role is its own name", "cublas_gemm|head" in v["batches"]["W2"]["names"]),
            ("the largest class difference leads", c2["by_class"][0]["key"] == "dense_gemm"
             and abs(c2["by_class"][0]["delta_ms"] - 1.0) < 1e-9),
            ("a name absent at W1 counts from zero", any(r["key"] == "_k25" and abs(r["delta_ms"] - 1.6) < 1e-9
                                                         for r in c2["by_name"])),
            ("a name gone at W2 reads negative", any(r["key"] == "_gemv_nf4_bw" and abs(r["delta_ms"] + 1.0) < 1e-9
                                                     for r in c2["by_name"])),
            ("shares are over the step delta", abs(c2["by_class"][0]["share_of_step_delta"] - round(1.0 / 2.01, 4)) < 1e-9),
            ("the summary renders", summary(v).startswith("CLIFF COMPLETE")),
        ]
    bad = [n for n, ok in cases if not ok]
    print(f"cliff_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    v = reduce(a.dir, a.e4b_sha)
    with open(a.out, "w") as f:
        json.dump(v, f, indent=1)
    print(summary(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
