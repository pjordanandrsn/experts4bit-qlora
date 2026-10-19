#!/usr/bin/env python3
"""bench/hp2/hp2_reduce.py -- HP2's registered reducer (bench/hp2/HP2-PREREG.md): the G1 gate and the verdict.

`g1`: from the one evaluator's calibration pass (hp1_eval.py over the two calibration arms, each on its own exported base, on R
calibration rows), the row-paired base differential delta0 = mean_r [g_e4b(r) - g_uns(r)], where g_f(r) is the evaluator's
zero-adapter loss on f's base minus f's in-loop step-0 loss on row r. Its null is delta0 = 0. CLOSE iff |delta0| + 2 x SE > 0.01;
STOP iff m_need = 2(|delta0| + 2 x SE) exceeds the expected margin (G2 would close); else PASS. It records the registered 6(b)
tolerance (each run's round trip, centred on its own zero-adapter gap z_f, at 0.01) and each framework's REPORTED band for z_f about
its calibration mean b_f, 3 x sd_f x sqrt(1/48 + 1/R) (never a VOID). Writes --out BEFORE
returning; exit 0 PASS, 4 CLOSE or STOP (a reading, not a strike; the box tears down), 2 unusable input.

`verdict`: from the confirmation pass (e4b K00 and Unsloth on seeds 3410-3412, each scored by the evaluator) and the G1 record:
the headline Delta (K00 - Unsloth, seed-paired, HP1's bands), the trained differential deltaN = mean_s [rho_e4b(s) - rho_uns(s)]
with its 90 % t-interval (rho = evaluator - in-loop step N, per run), and G2: deltaN's interval within +-m/2 and
|delta0| + 2 x SE <= m/2 for the realized m. Exit 0 on a read verdict, 3 otherwise (INCOMPLETE, VOID, CLOSED_G2).
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hp1"))   # in the repo
sys.path.insert(0, str(Path(__file__).resolve().parent))                 # on the box, the staged files sit side by side
import hp1_reduce as H1  # noqa: E402  (paired_reading, pooled_sd, order_problems, manifest_problems, t_ppf: tested in HP1)

G1_BOUND = 0.01          # half the 0.02 margin cap: no sigma_seed could rescue a wider instrument
G1_Z = 2.0               # G1 is a one-sided readability bound (maintainer ruling on #1793)
M_EXPECTED = 0.005       # no sigma_seed has been measured for this pair (TC1b's matched pair differed by 1e-4): the margin floor
VOID_Z = 3.0             # the reported z_f band: 3 x SE (z_f is deterministic per base, so a miss is reported, never a VOID)
ROUNDTRIP_MATCH = 0.01   # 6(b), zero-centred: the house's round-trip MATCH band (AXO1767-PREREG: MATCH <= 0.01)
SCORED_ROWS = 48
CAL_ROWS = 300           # as far as the fresh pool allows: held-out pool rows 97-396 (#1793 review: raise R)
CAL_TAGS = {"e4b": "hp2cal_e4b_k00_s3407", "unsloth": "hp2cal_unsloth_s3407"}
SEEDS = (3410, 3411, 3412)
EXIT_OK, EXIT_UNUSABLE, EXIT_INCOMPLETE, EXIT_G1_CLOSE = 0, 2, 3, 4

WORDING = {
    "BEATS": "e4b K00's held-out loss is lower than Unsloth's: the 90 % interval excludes 0 and |mean| > sigma_seed",
    "BEHIND": "e4b K00's held-out loss is higher than Unsloth's: the 90 % interval excludes 0 and |mean| > sigma_seed",
    "PARITY": "no measurable difference at margin m: the 90 % interval lies inside +-m",
    "INCONCLUSIVE": "no measurable effect, with its interval: neither inside +-m nor clear of 0",
    "CLOSED_G2": "the instrument could not read margin m (G2): no statement about the frameworks",
}


def _ceil6(x):
    return math.ceil(x * 1e6) / 1e6


def g1(ev, rows=CAL_ROWS):
    """The G1 record from an hp1_eval output dict. Raises ValueError on input that cannot be read as HP2's calibration."""
    gaps = {}
    for fw, tag in CAL_TAGS.items():
        recs = [b for b in ev.get("bases", []) if tag in (b.get("zero_adapter_row_gaps") or {})]
        if len(recs) != 1:
            raise ValueError(f"expected exactly one base with per-row zero-adapter gaps for {tag!r}, found {len(recs)}")
        if not recs[0].get("digest_matches_training_run"):
            raise ValueError(f"{fw}'s calibration base digest does not match its training run")
        g = recs[0]["zero_adapter_row_gaps"][tag]
        if len(g) != rows or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in g):
            raise ValueError(f"{fw}: expected {rows} finite per-row gaps, got {len(g)}")
        gaps[fw] = g
    d = [a - b for a, b in zip(gaps["e4b"], gaps["unsloth"])]
    delta0, sd_d = statistics.mean(d), statistics.stdev(d)
    se = sd_d / math.sqrt(rows)
    per, bands = {}, {}
    for fw, g in gaps.items():
        b, sd = statistics.mean(g), statistics.stdev(g)
        # REPORTED, never a VOID: where z_f (the base's zero-adapter gap on the 48 scored rows) should sit, 3 x the spread of a 48-row
        # mean about a rows-row calibration mean (SE n = 48 scored rows, with the calibration mean's own error over R rows)
        half = _ceil6(VOID_Z * sd * math.sqrt(1 / SCORED_ROWS + 1 / rows))
        per[fw] = {"b": round(b, 6), "sd": round(sd, 6), "mean_abs_gap": round(statistics.mean(abs(x) for x in g), 6),
                   "max_abs_gap": round(max(abs(x) for x in g), 6), "z_band_half_width": half}
        bands[fw] = [round(b, 6), half]
    reach = abs(delta0) + G1_Z * se
    m_need = 2 * reach
    out = {"reducer": "bench/hp2/hp2_reduce.py g1", "rows": rows, "delta0": round(delta0, 6), "sd_d": round(sd_d, 6),
           "se_delta0": round(se, 6), "null": 0.0, "reach": round(reach, 6), "bound": G1_BOUND, "z": G1_Z,
           "m_need": round(m_need, 6), "m_expected": M_EXPECTED, "per_framework": per, "z_bands": bands, "roundtrip_zero_centred_tol": ROUNDTRIP_MATCH,
           "void_z": VOID_Z, "false_void": (
               "6(b) is centred on each run's own zero-adapter gap, so it tests only the adapter round trip, at the registered "
               "0.01 (AXO1767's MATCH band; its e4b control read +0.0004). z_f - b_f is REPORTED against z_bands, never a VOID.")}
    if reach > G1_BOUND:
        out.update(verdict="CLOSE", reason=f"|delta0| + {G1_Z} x SE = {reach:.6f} > {G1_BOUND}: the instrument could not read margin "
                                           "0.02 (G1); a reading, not a strike")
    elif m_need > M_EXPECTED:
        out.update(verdict="STOP", reason=f"m_need = 2(|delta0| + {G1_Z} x SE) = {m_need:.6f} > the expected margin {M_EXPECTED}: G2 "
                                          "would close at the expected sigma_seed, so the box stops at G1 (a reading, not a strike)")
    else:
        out.update(verdict="PASS", reason=f"|delta0| + {G1_Z} x SE = {reach:.6f}; m_need {m_need:.6f} <= {M_EXPECTED}")
    return out


def _runs(ev):
    """{(framework, seed): arm} for HP2's confirmation tags; anything else is ignored."""
    out = {}
    for a in ev.get("arms", []):
        tag = a.get("tag") or ""
        for fw, prefix in (("e4b", "hp2_k00_s"), ("unsloth", "hp2_unsloth_s")):
            if tag.startswith(prefix) and tag[len(prefix):].isdigit():
                out[(fw, int(tag[len(prefix):]))] = a
    return out


def _delta_base(evs):
    """zero-adapter row mean on e4b's base minus on Unsloth's base, on the scored rows: reported beside Delta, so a BEATS or BEHIND
    can be attributed to training or to the base bytes. None when either base was not read."""
    z = {}
    for ev in evs:
        for b in ev.get("bases", []):
            name = str(b.get("dir", "")).rsplit("/", 1)[-1]
            for fw in ("e4b", "unsloth"):
                if name.startswith(f"base_{fw}_") and (b.get("zero_adapter") or {}).get("row_mean") is not None:
                    z[fw] = b["zero_adapter"]["row_mean"]
    return round(z["e4b"] - z["unsloth"], 6) if len(z) == 2 else None


def _z_vs_b(arms, g1rec):
    """Per framework: the runs' zero-adapter gaps z on the scored rows (one base, so one value up to numerics), z_f - b_f and whether
    it lies inside G1's reported band. Reported only."""
    out = {}
    for fw in ("e4b", "unsloth"):
        zs = [a["zero_gap"] for (f, _s), a in sorted(arms.items()) if f == fw and a.get("zero_gap") is not None]
        band = (g1rec.get("z_bands") or {}).get(fw)
        if not zs:
            out[fw] = None
            continue
        z = statistics.mean(zs)
        rec = {"z": zs, "z_f": round(z, 6)}
        if band:
            rec.update(b_f=band[0], z_minus_b=round(z - band[0], 6), band_half_width=band[1], inside=abs(z - band[0]) <= band[1])
        out[fw] = rec
    return out


def verdict(evs, g1rec):
    """The verdict record from the confirmation evaluator output(s) and the G1 record."""
    arms = {}
    for ev in evs:
        for k, a in _runs(ev).items():
            if k in arms:
                raise ValueError(f"duplicate run {k}")
            arms[k] = a
    ys, recs, rho, missing = {"e4b": {}, "unsloth": {}}, {"e4b": [], "unsloth": []}, {"e4b": {}, "unsloth": {}}, []
    for fw in ("e4b", "unsloth"):
        for s in SEEDS:
            a = arms.get((fw, s))
            ok = a is not None and a.get("status") == "SCORED" and (a.get("evaluator") or {}).get("row_mean") is not None
            if not ok:
                missing.append(f"{fw} seed {s}: {'absent' if a is None else a.get('status')}")
                continue
            ys[fw][s] = a["evaluator"]["row_mean"]
            rho[fw][s] = a.get("evaluator_minus_inloop_final")
            recs[fw].append({"tag": a["tag"], "seed": s, "order": a.get("order"), "own_base_only": bool(a.get("own_base_only"))})
    out = {"reducer": "bench/hp2/hp2_reduce.py verdict", "g1": {k: g1rec.get(k) for k in ("verdict", "delta0", "se_delta0", "m_need")},
           "y": {fw: {str(s): v for s, v in d.items()} for fw, d in ys.items()}, "missing": missing}
    out["delta_base"] = _delta_base(evs)       # the evaluator's zero-adapter difference on the 48 scored rows (#1793 review)
    out["z_vs_b"] = _z_vs_b(arms, g1rec)        # reported beside the zero-centred 6(b), never a VOID (#1793 ruling)
    if g1rec.get("verdict") != "PASS":
        out.update(verdict="G1_CLOSED", wording="the instrument could not read margin m (G1): no statement about the frameworks")
        return out, EXIT_INCOMPLETE
    if missing:
        out.update(verdict="INCOMPLETE", wording="a confirmation run has no evaluator number: no headline")
        return out, EXIT_INCOMPLETE
    probs = H1.order_problems(recs["e4b"] + recs["unsloth"])
    if probs:
        out.update(verdict="INCOMPLETE", order_problems=probs, wording="the runs do not share one data order per seed: no headline")
        return out, EXIT_INCOMPLETE
    sigma, sigma_df, _ = H1.pooled_sd([(f"P:{fw}", list(ys[fw].values())) for fw in ("e4b", "unsloth")])
    m = min(max(sigma, H1.MARGIN_FLOOR), H1.MARGIN_CAP)
    head = H1.paired_reading("e4b K00 vs Unsloth", ys["e4b"], recs["e4b"], ys["unsloth"], recs["unsloth"], sigma, m)
    e = [rho["e4b"][s] - rho["unsloth"][s] for s in SEEDS]
    k = len(e)
    dn, sd = statistics.mean(e), statistics.stdev(e)
    tq = H1.t_ppf(H1.CI_TAIL, k - 1)
    ci = [dn - tq * sd / math.sqrt(k), dn + tq * sd / math.sqrt(k)]
    g2_n = ci[0] >= -m / 2 and ci[1] <= m / 2
    g2_0 = (abs(g1rec["delta0"]) + G1_Z * g1rec["se_delta0"]) <= m / 2
    out.update(sigma_seed=sigma, sigma_seed_df=sigma_df, m=m, headline=head,
               deltaN={"mean": round(dn, 6), "ci90": [round(ci[0], 6), round(ci[1], 6)], "null": 0.0, "per_seed": e},
               g2={"deltaN_within_half_m": g2_n, "delta0_within_half_m": g2_0, "half_m": m / 2})
    if not (g2_n and g2_0):
        out.update(verdict="CLOSED_G2", wording=WORDING["CLOSED_G2"])
        return out, EXIT_INCOMPLETE
    out.update(verdict=head["verdict"], wording=WORDING[head["verdict"]])
    return out, EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("g1")
    a1.add_argument("eval_json")
    a1.add_argument("--rows", type=int, default=CAL_ROWS)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("verdict")
    a2.add_argument("evals", nargs="+")
    a2.add_argument("--g1", required=True)
    a2.add_argument("--manifest", action="append", default=[])
    a2.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "g1":
            rec = g1(json.load(open(a.eval_json)), a.rows)
            json.dump(rec, open(a.out, "w"), indent=1)                    # the record is written before the gate returns
            print(f"HP2_G1={rec['verdict']} delta0={rec['delta0']} se={rec['se_delta0']} m_need={rec['m_need']} "
                  f"rt_zero_centred={rec['roundtrip_zero_centred_tol']} z_bands={json.dumps(rec['z_bands'], sort_keys=True)}")
            return EXIT_OK if rec["verdict"] == "PASS" else EXIT_G1_CLOSE
        mprob = H1.manifest_problems(a.manifest, a.evals) if a.manifest else []
        if mprob:
            json.dump({"reducer": "bench/hp2/hp2_reduce.py verdict", "verdict": "VOID", "manifest_problems": mprob}, open(a.out, "w"), indent=1)
            print(f"HP2_VERDICT=VOID manifest: {mprob[:3]}")
            return EXIT_INCOMPLETE
        rec, code = verdict([json.load(open(p)) for p in a.evals], json.load(open(a.g1)))
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"HP2_VERDICT={rec['verdict']}: {rec.get('wording', '')}")
        return code
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        print(f"HP2 UNUSABLE {type(e).__name__}: {e}")
        return EXIT_UNUSABLE


if __name__ == "__main__":
    sys.exit(main())
