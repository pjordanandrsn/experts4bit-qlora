#!/usr/bin/env python3
"""Lane B771's reducer (bench/b771/PREREG-b771.md; e4b#771). Reads the two byte-stage reports (old and fixed
grouped-nf4-gemm) and the four decode receipts (A_old, P_old under the old kernel; P_new, A_new under the fixed
one) and applies the registered rule. The rule's quantities are computed HERE, from the receipts.

    python b771_reduce.py --dir <dir with bytes_{old,new}.json and b771_{A_old,P_old,P_new,A_new}.json> --out verdict.json
    python b771_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ARMS = ("A_old", "P_old", "P_new", "A_new")
MODES = {"A_old": "eager", "A_new": "eager", "P_old": "padded", "P_new": "padded"}
GNF4 = {"A_old": "old", "P_old": "old", "P_new": "new", "A_new": "new"}


def _first_divergence(x: dict, y: dict):
    for row in sorted(x, key=int):
        a, b = x[row], y.get(row, [])
        for i, (u, v) in enumerate(zip(a, b)):
            if u != v:
                return {"row": int(row), "token_index": i}
        if len(a) != len(b):
            return {"row": int(row), "token_index": min(len(a), len(b)), "length_differs": True}
    return None


def reduce(byt: dict, recs: dict) -> dict:
    """``byt`` maps 'old'/'new' -> byte-stage report; ``recs`` maps arm -> decode receipt."""
    v: dict = {"lane": "B771", "reasons": [], "verdict": None}
    for k in ("old", "new"):
        b = byt.get(k)
        if not isinstance(b, dict) or "payload_bytes_differing" not in b:
            v["verdict"] = "VOID"
            v["reasons"].append(f"byte stage '{k}' has no report")
            return v
    v["bytes"] = {k: {"gnf4": byt[k].get("gnf4"), "values": byt[k]["values"],
                      "payload_bytes_differing": byt[k]["payload_bytes_differing"],
                      "scales_differing": byt[k]["scales_differing"]} for k in ("old", "new")}
    if byt["old"].get("gnf4_has_e4m3_group") or not byt["new"].get("gnf4_has_e4m3_group"):
        v["verdict"] = "VOID"
        v["reasons"].append("the byte stages did not run the registered kernels (old must lack _e4m3_group, new have it)")
        return v
    missing = [k for k in ARMS if not isinstance(recs.get(k), dict) or "passes" not in recs[k]]
    if missing:
        v["verdict"] = "VOID"
        v["reasons"].append(f"decode arm(s) without a receipt: {missing}")
        return v
    for k in ARMS:
        r = recs[k]
        if r.get("mode") != MODES[k] or r.get("device_grouping") is not True:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k} ran mode {r.get('mode')!r} / device_grouping {r.get('device_grouping')!r}; "
                                f"registered {MODES[k]!r} with the device grouping")
            return v
        if r.get("gnf4_tag") != GNF4[k]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k} ran gnf4 {r.get('gnf4_tag')!r}, registered {GNF4[k]!r}")
            return v
        if not r.get("warm_equals_timed_tokens"):
            v["reasons"].append(f"{k}: warm and timed passes decoded different tokens (non-determinism)")
    trace = recs["A_old"]["trace"]
    if any(recs[k]["trace"] != trace for k in ARMS):
        v["verdict"] = "VOID"
        v["reasons"].append("arms disagree on the trace")
        return v
    tok = {k: recs[k]["passes"][1]["tokens"] for k in ARMS}
    idn = {"A_old_vs_A_new": _first_divergence(tok["A_old"], tok["A_new"]),
           "P_old_vs_A_old": _first_divergence(tok["P_old"], tok["A_old"]),
           "P_new_vs_A_new": _first_divergence(tok["P_new"], tok["A_new"]),
           "P_old_vs_P_new": _first_divergence(tok["P_old"], tok["P_new"])}
    v["identity"] = idn
    b_old, b_new = v["bytes"]["old"], v["bytes"]["new"]
    # the byte half: the fix must write the reference's bytes; the shipped kernel must be shown to differ
    if b_new["payload_bytes_differing"] or b_new["scales_differing"]:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"the fixed kernel still differs from quantize_kv_fp8: {b_new}")
        return v
    if idn["A_old_vs_A_new"] is not None:
        v["verdict"] = "VOID"
        v["reasons"].append(f"the control moved: eager decode changed with the gnf4 swap at {idn['A_old_vs_A_new']} "
                            "(the eager path never calls the fused append)")
        return v
    if idn["P_new_vs_A_new"] is not None:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"with the fix the bucket step still differs from the eager step at {idn['P_new_vs_A_new']}: "
                            "the fused append is not the whole cause")
        return v
    if b_old["payload_bytes_differing"] == 0 and b_old["scales_differing"] == 0:
        v["verdict"] = "VOID"
        v["reasons"].append("the shipped kernel wrote the reference's bytes on every value tested: nothing to explain")
        return v
    if idn["P_old_vs_A_old"] is None:
        v["verdict"] = "VOID"
        v["reasons"].append("under the shipped kernel the bucket step already equalled the eager step on this trace: "
                            "the divergence did not occur, so its removal cannot be shown")
        return v
    v["verdict"] = "CONFIRMED"
    v["reasons"].append(f"shipped kernel: {b_old['payload_bytes_differing']} payload / {b_old['scales_differing']} scale "
                        f"bytes differ over {b_old['values']} values and P_old != A_old at {idn['P_old_vs_A_old']}; fixed: "
                        f"0 / 0 over {b_new['values']} and P_new == A_new bitwise, A unchanged")
    return v


def _load(d: Path):
    byt, recs = {}, {}
    for k in ("old", "new"):
        f = d / f"bytes_{k}.json"
        if f.is_file():
            byt[k] = json.loads(f.read_text())
    for k in ARMS:
        for f in (d / f"b771_{k}.json", d / f"b771_{k}.json.gz"):
            if f.is_file():
                data = gzip.open(f).read() if f.suffix == ".gz" else f.read_bytes()
                recs[k] = json.loads(data)
                break
    return byt, recs


def _synthetic(*, old_bytes=5, new_bytes=0, p_old_tok=None, p_new_tok=None, a_new_tok=None):
    base = {"0": [1, 2, 3, 4], "1": [5, 6]}
    trace = [2, 2, 1, 1]
    byt = {"old": {"gnf4": "0.33.5", "gnf4_has_e4m3_group": False, "values": 1000,
                   "payload_bytes_differing": old_bytes, "scales_differing": 0},
           "new": {"gnf4": "0.33.7.dev", "gnf4_has_e4m3_group": True, "values": 1000,
                   "payload_bytes_differing": new_bytes, "scales_differing": 0}}
    toks = {"A_old": base, "P_old": p_old_tok or {"0": [1, 2, 9, 9], "1": [5, 6]},
            "P_new": p_new_tok or base, "A_new": a_new_tok or base}
    recs = {k: {"mode": MODES[k], "device_grouping": True, "gnf4_tag": GNF4[k], "trace": trace,
                "warm_equals_timed_tokens": True, "passes": [{"tokens": toks[k]}, {"tokens": toks[k]}]}
            for k in ARMS}
    return byt, recs


def self_test() -> None:
    assert reduce(*_synthetic())["verdict"] == "CONFIRMED"
    # the fix leaves a byte different: refuted
    assert reduce(*_synthetic(new_bytes=1))["verdict"] == "REFUTED"
    # with the fix the bucket step still differs: not the whole cause
    got = reduce(*_synthetic(p_new_tok={"0": [1, 2, 3, 7], "1": [5, 6]}))
    assert got["verdict"] == "REFUTED" and got["identity"]["P_new_vs_A_new"] == {"row": 0, "token_index": 3}, got
    # the control moved: void
    assert reduce(*_synthetic(a_new_tok={"0": [1, 2, 3, 8], "1": [5, 6]}))["verdict"] == "VOID"
    # nothing to explain: the old kernel matched, or the divergence did not occur on the trace
    assert reduce(*_synthetic(old_bytes=0))["verdict"] == "VOID"
    assert reduce(*_synthetic(p_old_tok={"0": [1, 2, 3, 4], "1": [5, 6]}))["verdict"] == "VOID"
    # a missing arm, or the wrong kernel in a byte stage, voids
    b, r = _synthetic()
    del r["P_new"]
    assert reduce(b, r)["verdict"] == "VOID"
    b, r = _synthetic()
    b["new"]["gnf4_has_e4m3_group"] = False
    assert reduce(b, r)["verdict"] == "VOID"
    b, r = _synthetic()
    r["A_old"]["device_grouping"] = False
    assert reduce(b, r)["verdict"] == "VOID"
    print("b771_reduce self-test OK (9 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(*_load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"B771_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
