# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-t5-dispatch-diet verdict calculator. Bars are hardcoded from the
prereg; the runner feeds files, the calculator prints the verdict. Run with
--self-test FIRST (both directions) before any real receipt touches it.

Inputs: four timed reps (aa1 aa2=A1 b a3; step_decomp --out JSONs) and two
sync-attr JSONs (--aprof/--bprof from --sync-attr-out). Optional
--amort-side rep reports the instrument tax (no bar)."""

import argparse
import json
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5          # G0: half the primary bar
H1_ENGINE_FRAC = 0.50        # attribution floor on aten::nonzero
H2_NONZERO_PER_STEP = 60.0   # from 174/step
H2_CHURN_CUT = 15.0          # % cut in copy_+to+_to_copy+index_select
H3_PASS = 15.0               # % median-step improvement, B vs min(A1, A3)
H3_PARTIAL = 8.0


def _med(rep):
    return float(rep["decode_median_ms"]["step"])


def _tokens(rep):
    return json.dumps(rep["generated_tokens"], sort_keys=True)


def _churn(prof):
    oc = prof["op_counts"]
    return sum(oc.get(k, 0) for k in
               ("aten::copy_", "aten::to", "aten::_to_copy",
                "aten::index_select"))


def verdict(aa1, aa2, b, a3, aprof, bprof, amort_side=None):
    out = {}
    m1, m2, mb, m3 = _med(aa1), _med(aa2), _med(b), _med(a3)
    spread = abs(m1 - m2) / min(m1, m2) * 100.0
    out["g0"] = {"aa1_ms": m1, "aa2_ms": m2, "spread_pct": spread,
                 "pass": spread < AA_SPREAD_MAX}
    if not out["g0"]["pass"]:
        out["verdict"] = "NO-VERDICT (G0 fail: box not measurement-grade)"
        return out

    idents = {_tokens(r) for r in (aa1, aa2, b, a3)}
    out["g1"] = {"identical": len(idents) == 1, "n_distinct": len(idents)}
    if not out["g1"]["identical"]:
        out["verdict"] = "REFUTED (G1: the diet changed decoded tokens)"
        return out

    for name, prof in (("aprof", aprof), ("bprof", bprof)):
        if prof["active_steps"] != 12:
            out["verdict"] = (f"NO-VERDICT ({name} active window "
                              f"{prof['active_steps']}/12 incomplete)")
            return out
    assert not aprof["dispatch_diet"] and bprof["dispatch_diet"], \
        "profile arms mislabeled: aprof must be baseline, bprof diet"

    nz = aprof["nonzero_attr"]
    tot_nz = nz["engine"] + nz["other"]
    frac = nz["engine"] / tot_nz if tot_nz else 0.0
    out["h1"] = {"engine_frac": frac, "nonzero_total": tot_nz,
                 "pass": frac >= H1_ENGINE_FRAC}
    if not out["h1"]["pass"]:
        out["verdict"] = ("STOP-AND-AMEND (H1: registered edits aim at the "
                          "wrong sites; re-register before any B claim)")
        return out

    steps = bprof["active_steps"]
    nz_b = bprof["op_counts"].get("aten::nonzero", 0) / steps
    churn_a, churn_b = _churn(aprof), _churn(bprof)
    churn_cut = (churn_a - churn_b) / churn_a * 100.0 if churn_a else 0.0
    out["h2"] = {"nonzero_per_step_b": nz_b,
                 "nonzero_per_step_a":
                     aprof["op_counts"].get("aten::nonzero", 0) / 12,
                 "churn_a": churn_a, "churn_b": churn_b,
                 "churn_cut_pct": churn_cut,
                 "pass": (nz_b <= H2_NONZERO_PER_STEP
                          and churn_cut >= H2_CHURN_CUT)}

    base = min(m2, m3)
    dpct = (base - mb) / base * 100.0
    out["h3"] = {"base_ms": base, "b_ms": mb, "improvement_pct": dpct,
                 "pass": dpct >= H3_PASS,
                 "partial": H3_PARTIAL <= dpct < H3_PASS}

    h2, h3 = out["h2"]["pass"], out["h3"]
    if h2 and h3["pass"]:
        out["verdict"] = "CERTIFIED (flip dispatch_diet default on)"
    elif h2 and h3["partial"]:
        out["verdict"] = ("PARTIAL (ship only if no op-count regression "
                          "anywhere in the table; record honestly)")
    elif h2:
        out["verdict"] = ("REFUTED (syncs were not the cost; revert -- "
                          "ladder re-points at host busy-work)")
    elif h3["pass"]:
        out["verdict"] = ("CERTIFIED-WITH-OPEN-MECHANISM (wall win real, "
                          "mis-explained; ship and file the gap)")
    else:
        out["verdict"] = "REFUTED (revert)"

    if amort_side is not None:
        ms = _med(amort_side)
        out["amort_tax_reported"] = {
            "amort_on_ms": ms, "amort_off_ms": m2,
            "tax_pct": (ms - m2) / m2 * 100.0}
    return out


def _fake_rep(ms, tok="x"):
    return {"decode_median_ms": {"step": ms},
            "generated_tokens": {"0": [1, 2, tok]}}


def _fake_prof(nonzero, churn, diet, engine=None, active=12):
    return {"active_steps": active, "dispatch_diet": diet,
            "op_counts": {"aten::nonzero": nonzero, "aten::copy_": churn,
                          "aten::to": 0, "aten::_to_copy": 0,
                          "aten::index_select": 0},
            "nonzero_attr": {"engine": engine if engine is not None
                             else nonzero, "other": nonzero - (
                                 engine if engine is not None else nonzero),
                             "frames": {}}}


def self_test():
    # direction 1: a clean pass must CERTIFY
    v = verdict(_fake_rep(140.0), _fake_rep(139.0), _fake_rep(110.0),
                _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False),
                _fake_prof(40 * 12, 60000, True))
    assert v["verdict"].startswith("CERTIFIED ("), v
    # direction 2: no wall win must REFUTE even with clean mechanism
    v = verdict(_fake_rep(140.0), _fake_rep(139.0), _fake_rep(136.0),
                _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False),
                _fake_prof(40 * 12, 60000, True))
    assert v["verdict"].startswith("REFUTED (syncs were not"), v
    # G1 breach dominates everything
    v = verdict(_fake_rep(140.0), _fake_rep(139.0),
                _fake_rep(110.0, tok="DIFFERENT"), _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False),
                _fake_prof(40 * 12, 60000, True))
    assert v["verdict"].startswith("REFUTED (G1"), v
    # G0 breach blocks any verdict
    v = verdict(_fake_rep(160.0), _fake_rep(139.0), _fake_rep(110.0),
                _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False),
                _fake_prof(40 * 12, 60000, True))
    assert v["verdict"].startswith("NO-VERDICT (G0"), v
    # H1 miss stops-and-amends
    v = verdict(_fake_rep(140.0), _fake_rep(139.0), _fake_rep(110.0),
                _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False, engine=100),
                _fake_prof(40 * 12, 60000, True))
    assert v["verdict"].startswith("STOP-AND-AMEND"), v
    # partial band
    v = verdict(_fake_rep(140.0), _fake_rep(139.0), _fake_rep(125.5),
                _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False),
                _fake_prof(40 * 12, 60000, True))
    assert v["verdict"].startswith("PARTIAL"), v
    # wall win with dirty mechanism = certified-with-open-mechanism
    v = verdict(_fake_rep(140.0), _fake_rep(139.0), _fake_rep(110.0),
                _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False),
                _fake_prof(120 * 12, 89000, True))
    assert v["verdict"].startswith("CERTIFIED-WITH-OPEN"), v
    # incomplete profile window blocks
    v = verdict(_fake_rep(140.0), _fake_rep(139.0), _fake_rep(110.0),
                _fake_rep(140.5),
                _fake_prof(174 * 12, 90000, False, active=9),
                _fake_prof(40 * 12, 60000, True))
    assert v["verdict"].startswith("NO-VERDICT (aprof"), v
    print("self-test OK: 8/8 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--aa1")
    ap.add_argument("--aa2")
    ap.add_argument("--b")
    ap.add_argument("--a3")
    ap.add_argument("--aprof")
    ap.add_argument("--bprof")
    ap.add_argument("--amort-side", default=None)
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    v = verdict(load(a.aa1), load(a.aa2), load(a.b), load(a.a3),
                load(a.aprof), load(a.bprof),
                load(a.amort_side) if a.amort_side else None)
    print(json.dumps(v, indent=2))
