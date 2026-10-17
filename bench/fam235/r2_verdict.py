# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2-Mixtral's milestone verdict (bench/fam235/PREREG-r2-mixtral.md, Amendment 1), applied mechanically, never read by eye.

The decision rules, the VOID causes and Amendment 3's instrument checks are Q235's (``bench/q235/instrument/q235_verdict.py`` at the
commit R2 cites, ``r2_drift.CITED``), run unchanged, by citation. Two things differ.

**The band.** Q235's single 0.775 is replaced by Amendment 1's depth-local band, one threshold used both ways:

    band_j = 2 x u_cal(m(j)),   m(j) = round(32 j / n)

Here u_cal is the Mixtral-8x7B calibration's unplanted box u at layer m(j), read from ``r2-mixtral-a1-numbers.json`` (pinned
by sha256).
- **Plants:** each plant must read u_L > band_L at its layer L = n // 2. Otherwise the box is VOID (instrument), as in Q235.
- **Rule 7:** the unplanted box must read u_j <= band_j on every interior layer. Otherwise it is a FAIL.
- **The map:** a tie (32 j / n exactly k + 1/2) is refused, never resolved by a rounding mode. On 56 layers there is none.

**The registered class and estimates.** ``REGISTERED`` holds the card class and the device and link estimates. The amendment that
registers the box pins them. Until then any science run is VOID ("no box is registered"), so no placeholder value is ever read.

**Reported beside Q235's report:**
- each interior layer's margin band_j / u_j, the 8x7B layer it maps to, and the smallest margin and where it falls;
- each plant's u_L / band_L;
- the shallow band, named as where a noise FAIL is expected. On new 8x7B tokens the band's least slack was at 8x7B layers 3-8
  (2.006-2.064, against 2.056-2.366 deeper); on 56 layers those are j = 5..14.

Exit codes: PASS 0, FAIL 1, VOID 2. ``verdict.json`` holds every number the verdict used.

    python r2_verdict.py RUN_DIR --ref REF_DIR --out VERDICT.json
    python r2_verdict.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
NUMBERS = os.path.join(HERE, "r2-mixtral-a1-numbers.json")
NUMBERS_SHA256 = "5d2bac26b0c5e5dd4c29962d21a95ac3be8292d34322189012b0ca2866d41279"    # as merged in e4b#1643
N_CAL = 32                                     # Mixtral-8x7B's decoder layers
SHALLOW_CAL = (3, 8)                           # the 8x7B layers with the validation's smallest band/u (2.006-2.064)
REGISTERED = None                              # the box-registering amendment pins {"card_re", "estimate_device_gib", "estimate_link_gib"}
EXIT = {"PASS": 0, "FAIL": 1, "VOID": 2}


def q235_verdict():
    """Q235's verdict tool at the commit R2 cites (``r2_drift.cited_tree``), loaded unchanged as a private module instance (its
    constants are set per call, never on disk). Its siblings (the gate-1 repeat rule, the box's STRICT rules, the compare) load from
    the same cited tree."""
    drift_spec = importlib.util.spec_from_file_location("r2_drift_for_verdict", os.path.join(HERE, "r2_drift.py"))
    drift = importlib.util.module_from_spec(drift_spec)
    drift_spec.loader.exec_module(drift)
    spec = importlib.util.spec_from_file_location("q235_verdict_for_r2",
                                                  str(drift.cited_tree() / "bench" / "q235" / "instrument" / "q235_verdict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def numbers():
    raw = open(NUMBERS, "rb").read()
    if hashlib.sha256(raw).hexdigest() != NUMBERS_SHA256:
        raise ValueError(f"{os.path.basename(NUMBERS)} is not the file Amendment 1 registered (sha256 mismatch)")
    return json.loads(raw)


def calibrated_band():
    """Amendment 1's band on the 8x7B layers, {8x7B layer: 2 x u_cal}."""
    return {int(i): b for i, b in numbers()["depth_local"]["band"].items()}


def depth_map(j, n):
    """m(j) = round(32 j / n), exactly; a tie is refused rather than resolved."""
    x = Fraction(N_CAL * j, n)
    if x.denominator == 2:
        raise ValueError(f"layer {j} of {n} maps to {float(x)} on the 8x7B: a tie, which Amendment 1 does not resolve")
    return round(x)


def band_profile(n):
    """{j: (8x7B layer, band_j)} on n layers' interior class, with the plant layer n // 2 among them."""
    cal, interior = calibrated_band(), range(-(-4 * n // 48), 44 * n // 48 + 1)
    out = {}
    for j in sorted(set(interior) | {n // 2}):
        m = depth_map(j, n)
        if m not in cal:
            raise ValueError(f"layer {j} of {n} maps to 8x7B layer {m}, outside the calibrated interior {min(cal)}..{max(cal)}")
        out[j] = (m, cal[m])
    return out


def read_band(u, n):
    """Rule 7 and its report, from the unplanted box's u profile (``q235_compare``'s ``update_rel_err``)."""
    prof, interior = band_profile(n), range(-(-4 * n // 48), 44 * n // 48 + 1)
    over = [j for j in interior if not u[j] <= prof[j][1]]
    margin = {j: (prof[j][1] / u[j] if u[j] else math.inf) for j in interior}
    jmin = min(margin, key=margin.get)
    shallow = [j for j in interior if SHALLOW_CAL[0] <= prof[j][0] <= SHALLOW_CAL[1]]
    return {"ok": not over, "interior": [interior[0], interior[-1]], "layers_over": over,
            "band": {j: prof[j][1] for j in interior}, "maps_to": {j: prof[j][0] for j in interior},
            "margin": margin, "margin_min": margin[jmin], "margin_min_layer": jmin,
            "shallow_band": [shallow[0], shallow[-1]] if shallow else None,
            "layers_over_in_shallow_band": [j for j in over if j in shallow],
            "u_interior_max": max(u[j] for j in interior)}


def instrument_cause(rec, n, repeat_dir):
    """Amendment 3's instrument checks in Q235's order, with band_L in place of Q235's single band: the null, then each plant's
    position, earlier layers, engagement and band. Updates each plant's record; returns the first cause, or None."""
    L, band_L = n // 2, band_profile(n)[n // 2][1]
    cause = None if rec["null_vs_box_max"] <= rec["null_floor"] else \
        f"the null is not the box (max {rec['null_vs_box_max']:.3g} > {rec['null_floor']})"
    for kind, p in rec["plants"].items():
        placed = json.load(open(os.path.join(repeat_dir, f"plant_{kind}", "box.json")))["plant"]
        p.update(band_at_layer=band_L, fails_band=p["u_at_layer"] > band_L, u_over_band=p["u_at_layer"] / band_L)
        why = (f"the {kind} plant is not at layer {L}" if placed != {"kind": kind, "layer": L} else
               f"the {kind} plant touched the layers before it (max {p['pre_plant_max_rel_diff']:.3g} > {rec['null_floor']})"
               if not p["pre_plant_within_floor"] else
               f"the {kind} plant did not engage" if not p["planted_layer_differs"] else
               f"the {kind} plant passes the band (u {p['u_at_layer']:.3g} <= band_{L} {band_L:.4g}): the band cannot see it"
               if not p["fails_band"] else None)
        cause = cause or why
    return cause


def judge(run_dir, ref_dir):
    if REGISTERED is None:
        return {"verdict": "VOID", "why": "no box is registered: the registering amendment pins the class and estimates",
                "rules": {}, "reported": {}}
    qv = q235_verdict()
    qv.CARD_RE, qv.ESTIMATE_DEVICE_GIB, qv.ESTIMATE_LINK_GIB = (
        REGISTERED["card_re"], REGISTERED["estimate_device_gib"], REGISTERED["estimate_link_gib"])
    qv.BAND = -math.inf                     # Q235's single band is not R2's: every band comparison is re-read below
    out = qv.judge(run_dir, ref_dir)
    out["tool"] = "r2_verdict: Q235's verdict with Amendment 1's band"
    r = out.get("rules") or {}
    if not r:                               # a VOID before any rule was read (refusal, host fault, guard, harness error)
        return out
    u = out["reported"]["u_profile"]
    n = len(u)
    stopped = out.get("exit_code") == "12"  # the STRICT failure stops the run before the repeat: rules 6 and 7 not read
    instrument = None
    try:
        if not stopped:
            r["7_band"] = read_band(u, n)
            instrument = instrument_cause(out["instrument"], n, os.path.join(run_dir, "receipts", "repeat"))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        out.update(verdict="VOID", why=f"harness error: {type(e).__name__}: {e}")
        return out
    failed = [k for k, v in r.items() if v["ok"] is False]
    if instrument:
        out.update(verdict="VOID", why=f"instrument: {instrument}" + (f" (rules also failing: {', '.join(failed)})" if failed else ""))
    else:
        out.update(verdict="FAIL" if failed else "PASS", why=("rules failing: " + ", ".join(failed)) if failed else "rules 1-7 hold")
    return out


def self_test():
    """Amendment 1's numbers, replayed: the map's properties, both 8x7B reads under the identity map, and the refusals."""
    d = numbers()
    cal = calibrated_band()
    checks = {}
    p56 = band_profile(56)
    checks["56: interior 5..51 maps onto exactly 3..29"] = (min(p56) == 5 and max(p56) == 51
                                                            and sorted({m for m, _ in p56.values()}) == list(range(3, 30)))
    checks["56: L28 maps to L16, band_28 = 0.47432"] = p56[28] == (16, cal[16]) and abs(cal[16] - 0.47432) < 1e-12
    checks["56: no ties on any layer"] = all(Fraction(N_CAL * j, 56).denominator != 2 for j in range(56))
    checks["56: the shallow band is j = 5..14"] = read_band([1.0] * 56, 56)["shallow_band"] == [5, 14]
    checks["32: the map is the identity"] = all(m == j for j, (m, _) in band_profile(32).items())
    for read, want_min in (("validation", (2.006, 3)), ("calibration", (2.0, None))):
        u = d[read]["u"]
        b = read_band(u["box"], 32)
        plants = all(u[p][16] > cal[16] for p in ("plant_routing", "plant_epilogue"))
        checks[f"{read}: line F holds"] = b["ok"]
        checks[f"{read}: line S holds"] = plants
        checks[f"{read}: smallest margin {want_min[0]}"] = (abs(b["margin_min"] - want_min[0]) < 5e-4
                                                           and (want_min[1] is None or b["margin_min_layer"] == want_min[1]))
    v = d["depth_local"]["validation"]
    checks["validation: VALIDATED, as recorded"] = (v["verdict"] == "VALIDATED" and checks["validation: line F holds"]
                                                    and checks["validation: line S holds"]
                                                    and read_band(d["validation"]["u"]["box"], 32)["margin_min"] == v["line_F_margin_min"])
    checks["calibration: the withdrawn global form misses the routing plant"] = d["calibration"]["u"]["plant_routing"][16] < 2 * \
        d["global_form"]["U"]
    for n, bad in ((64, "a tie"),):
        try:
            band_profile(n)
            checks[f"{n}: refused ({bad})"] = False
        except ValueError as e:
            checks[f"{n}: refused ({bad})"] = "tie" in str(e)
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not (a.run_dir and a.ref and a.out):
        ap.error("RUN_DIR, --ref and --out are required")
    r = judge(a.run_dir, a.ref)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(json.dumps({"verdict": r["verdict"], "why": r["why"]}))
    sys.exit(EXIT[r["verdict"]])


if __name__ == "__main__":
    main()
