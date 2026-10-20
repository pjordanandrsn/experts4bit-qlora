#!/usr/bin/env python3
"""bench/hpn/hpn_reduce.py -- HPN's decomposition (bench/hpn/HPN-DESIGN.md): which side carries the per-row gap.

At each depth L, on one exported base and the same rows, three readers give per-row losses, each run twice:
  I(r)  the framework's in-loop step-0 loss (two independent one-step arms; tc1 receipts' hp1.inloop_step0_rows);
  E(r)  the one evaluator, bf16 (two hp1_eval.py outputs; bases[0].zero_adapter.row_losses);
  F(r)  the fp32 reference (hpn_read.py's two runs).
Stage 1b (Unsloth's readers at each depth) is read first: MOOT if |delta0_F| / |delta0_E| > 0.93 at a deciding depth,
INDETERMINATE if |delta0_E| < 2 SE there. With g = E - I, e = E - F and i = I - F (run 1 of each), the side share
var(e) / (var(e) + var(i)) reads E-dominant >= 2/3,
I-dominant <= 1/3, comparable otherwise. The outcome follows the note's table: STAGE2 only if E-dominant at the two deepest depths;
STOP otherwise (DEPTH_DEPENDENT if the two deepest calls differ). Any 48-layer figure is a log-log EXTRAPOLATION of sd against
depth, with its leave-one-out range. Never a scored number, and never a statement about either framework.

    python hpn_reduce.py SPEC.json --out RESULT.json
SPEC = {"framework": "e4b", "free_mib_at_claim": N, "depths": [{"L": 2, "inloop": [R1, R2], "eval": [E1, E2], "fref": F,
        "unsloth": {"inloop": [RU], "eval": [EU], "fref": FU}}, ...]}  (the unsloth group is stage 1b: one run of each reader)
"""
import argparse
import json
import math
import os
import statistics
import sys

SHARE_HI, SHARE_LO = 2 / 3, 1 / 3      # E-dominant at >= 2/3, I-dominant at <= 1/3 (a variance ratio of 2 either way)
N0_TOL = 1e-5 + 1e-9                    # two runs agree per row to 1e-5 (per-row losses are recorded to 5 decimals)
N1_L2_BOUND = 0.03                      # N1: sd(g) at L2 below HP2's 48-layer ~0.03
N3_RHO = 0.2                            # N3: |corr(e, i)| <= 0.2 at every fitted depth
DEPTH_FLOOR = 8                         # the maintainer's ruling: the decision needs the deepest fitted depth >= L8
# stage 1b, scale-free: at each deciding depth stage 2 needs |delta0_F(L)| / |delta0_E(L)| <= 0.0025 / 0.002691 -- G1's half-floor
# against HP2's 48-layer |delta0_E| -- else MOOT; a depth where |delta0_E| < 2 SE (G1's 2 SE) is INDETERMINATE, never a pass
MOOT_RATIO = round(0.0025 / 0.002691, 2)
GUARD_Z = 2.0
EXTRAP_L = 48


def _inloop_rows(path):
    """The receipt's step-0 per-row losses, read the way the evaluator reads them (hp1_eval.receipt_arm: eval_curve's step 0)."""
    curve = json.load(open(path)).get("eval_curve") or []
    rows = [c.get("row_losses") for c in curve if c.get("step") == 0]
    if not rows or not rows[0]:
        raise ValueError(f"{path}: no in-loop step-0 per-row losses")
    return [float(x) for x in rows[0]]


def _eval_rows(path):
    ev = json.load(open(path))
    bases = [b for b in ev.get("bases", []) if (b.get("zero_adapter") or {}).get("row_losses")]
    if len(bases) != 1:
        raise ValueError(f"{path}: expected exactly one base with zero-adapter row losses, found {len(bases)}")
    if not bases[0].get("digest_matches_training_run"):
        raise ValueError(f"{path}: the base's digest does not match its training run")
    return [float(x) for x in bases[0]["zero_adapter"]["row_losses"]]


def _fref_rows(path):
    f = json.load(open(path))
    if f.get("reader") != "F" or len(f.get("runs") or []) < 2:
        raise ValueError(f"{path}: not an hpn_read.py output with two runs")
    return [[float(x) for x in run] for run in f["runs"][:2]]


def _corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / ((len(a) - 1) * sa * sb)


def _stat(v):
    return {"mean": round(statistics.mean(v), 6), "sd": round(statistics.stdev(v), 6)}


def _call(share):
    return "E" if share >= SHARE_HI else "I" if share <= SHARE_LO else "comparable"


def depth_record(In, E, Fr):
    """One depth from two runs per reader: N0 per reader, the decomposition on run 1, the side call."""
    n = len(In[0])
    if not all(len(x) == n for x in (In[1], E[0], E[1], Fr[0], Fr[1])) or n < 3:
        raise ValueError(f"readers disagree on the row count (or fewer than 3 rows): {[len(x) for x in (*In, *E, *Fr)]}")
    n0 = {name: max(abs(a - b) for a, b in zip(*runs)) for name, runs in (("I", In), ("E", E), ("F", Fr))}   # unrounded: compared below
    g = [a - b for a, b in zip(E[0], In[0])]
    e = [a - b for a, b in zip(E[0], Fr[0])]
    i = [a - b for a, b in zip(In[0], Fr[0])]
    ve, vi, vg = statistics.variance(e), statistics.variance(i), statistics.variance(g)
    share = ve / (ve + vi) if (ve + vi) > 0 else None
    rho = _corr(e, i)
    n3 = rho is None or abs(rho) <= N3_RHO                 # e or i constant: no correlation to condition on
    # where N3 fails, the same thresholds apply to what swapping E for F would leave: 1 - var(i) / var(g) (= share when uncorrelated)
    basis = share if n3 else (1 - vi / vg if vg > 0 else None)
    return {"rows": n, "g": _stat(g), "e": _stat(e), "i": _stat(i), "share_e": None if share is None else round(share, 4),
            "call_basis": "share" if n3 else "1 - var(i)/var(g)", "call_value": None if basis is None else round(basis, 4),
            "call": None if basis is None else _call(basis), "corr_e_i": None if rho is None else round(rho, 4),
            "n0_max_abs_rerun": {k: round(v, 6) for k, v in n0.items()}, "n0_holds": {k: v <= N0_TOL for k, v in n0.items()},
            "n3_holds": n3, "_sd_g": statistics.stdev(g)}            # every bar reads the UNROUNDED value (review); _sd_g feeds N1


def _fit(depths, sds):
    """log(sd) = a + p log(L) by least squares; (prediction at EXTRAP_L, slope p) or None."""
    pts = [(math.log(L), math.log(s)) for L, s in zip(depths, sds) if s and s > 0]
    if len(pts) < 2:
        return None
    mx, my = statistics.mean(x for x, _ in pts), statistics.mean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    p = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return math.exp(my + p * (math.log(EXTRAP_L) - mx)), p


def extrapolation(depths, sds):
    """The 48-layer figure for one sd series, labelled an EXTRAPOLATION: the full fit and its leave-one-out range (>= 3 depths)."""
    full = _fit(depths, sds)
    if full is None:
        return None
    loo = [f[0] for k in range(len(depths)) if len(depths) >= 3
           for f in [_fit(depths[:k] + depths[k + 1:], sds[:k] + sds[k + 1:])] if f is not None]
    return {"label": "EXTRAPOLATION (log-log fit of sd against depth; not a measurement of the 48-layer base)",
            "at_L": EXTRAP_L, "value": round(full[0], 6), "slope": round(full[1], 4),
            "loo_range": [round(min(loo), 6), round(max(loo), 6)] if loo else None, "fitted_depths": depths}


def base_term(e4b, uns):
    """Stage 1b at one depth: delta0 = (base term) - (in-loop term), under E and under F (run 1 of each reader; same rows), with
    the per-row differentials d_X(r) = (X_e4b - I_e4b)(r) - (X_uns - I_uns)(r) behind each SE, the ratio and the 1b read."""
    m = statistics.mean
    n = len(e4b["I"])
    inloop = m(e4b["I"]) - m(uns["I"])
    bE, bF = m(e4b["E"]) - m(uns["E"]), m(e4b["F"]) - m(uns["F"])
    d = {x: [(a - i) - (b - j) for a, i, b, j in zip(e4b[x], e4b["I"], uns[x], uns["I"])] for x in ("E", "F")}
    dE, dF = bE - inloop, bF - inloop
    seE, seF = statistics.stdev(d["E"]) / math.sqrt(n), statistics.stdev(d["F"]) / math.sqrt(n)
    se_shift = statistics.stdev([f - e for e, f in zip(d["E"], d["F"])]) / math.sqrt(n)
    readable = abs(dE) >= GUARD_Z * seE
    ratio = abs(dF) / abs(dE) if dE else None
    shifted = abs(dF - dE) > GUARD_Z * se_shift
    # OPEN needs a ratio <= 0.93 that is also a measurable shift; a ratio inside the shift's own noise is INDETERMINATE (ruled)
    read = ("INDETERMINATE" if not readable or ratio is None else "MOOT" if ratio > MOOT_RATIO else "OPEN" if shifted
            else "INDETERMINATE")
    return {"inloop_term": round(inloop, 6), "base_term_E": round(bE, 6), "base_term_F": round(bF, 6),
            "shift_E_to_F": round(bF - bE, 6), "se_shift": round(se_shift, 6),
            "delta0_E": round(dE, 6), "se_delta0_E": round(seE, 6), "abs_delta0_E": round(abs(dE), 6),
            "delta0_F": round(dF, 6), "se_delta0_F": round(seF, 6), "abs_delta0_F": round(abs(dF), 6),
            "ratio_F_over_E": None if ratio is None else round(ratio, 4), "read_1b": read,
            # a SHIFT E -> F is called only past 2 x its paired SE; otherwise "no measurable shift", with its interval (reported only)
            "shift_call": "SHIFT" if shifted else "no measurable shift",
            "shift_interval": [round(dF - dE - GUARD_Z * se_shift, 6), round(dF - dE + GUARD_Z * se_shift, 6)]}


def reduce(spec, load=None):
    load = load or {"inloop": _inloop_rows, "eval": _eval_rows, "fref": _fref_rows}
    recs, bterm = {}, {}
    for d in sorted(spec["depths"], key=lambda x: int(x["L"])):
        In = [load["inloop"](p) for p in d["inloop"]]
        E = [load["eval"](p) for p in d["eval"]]
        Fr = load["fref"](d["fref"])
        recs[int(d["L"])] = depth_record(In, E, Fr)
        u = d.get("unsloth")
        if u:
            ui, ue, uf = load["inloop"](u["inloop"][0]), load["eval"](u["eval"][0]), load["fref"](u["fref"])[0]
            if not len(ui) == len(ue) == len(uf) == len(In[0]):
                raise ValueError(f"L{d['L']}: Unsloth's readers and e4b's disagree on the row count")
            bterm[int(d["L"])] = base_term({"I": In[0], "E": E[0], "F": Fr[0]}, {"I": ui, "E": ue, "F": uf})
    Ls = sorted(recs)
    out = {"reducer": "bench/hpn/hpn_reduce.py", "framework": spec.get("framework"), "free_mib_at_claim": spec.get("free_mib_at_claim"),
           "climb_stop": spec.get("climb_stop") or None,
           "fitted_depths": Ls, "deepest_fit": Ls[-1] if Ls else None,
           "depths": {str(L): {k: v for k, v in recs[L].items() if not k.startswith("_")} for L in Ls},
           "thresholds": {"share_e_dominant": round(SHARE_HI, 4), "share_i_dominant": round(SHARE_LO, 4), "n0_tol": 1e-5,
                          "n1_l2_bound": N1_L2_BOUND, "n3_rho": N3_RHO, "depth_floor": DEPTH_FLOOR,
                          "moot_ratio": MOOT_RATIO, "guard_z": GUARD_Z},
           "base_term": {str(L): bterm[L] for L in sorted(bterm)}}
    if len(Ls) < 2:
        out.update(outcome="INSUFFICIENT", reason="fewer than two depths fit: N2's two-deepest rule cannot be read")
        return out
    if Ls[-1] < DEPTH_FLOOR:                                         # the driver's own reason for stopping the climb, when it gave one
        why = spec.get("climb_stop") or "wait for the sidecar to unload and retry"
        out.update(outcome="INSUFFICIENT_DEPTH", reason=f"the deepest fitted depth L{Ls[-1]} is below the L{DEPTH_FLOOR} floor: {why}")
        return out
    out["N0"] = {k: all(recs[L]["n0_holds"][k] for L in Ls) for k in ("I", "E", "F")}
    raw = [recs[L]["_sd_g"] for L in Ls]
    rises = all(b > a for a, b in zip(raw, raw[1:]))
    out["N1"] = None if 2 not in recs else (rises and recs[2]["_sd_g"] < N1_L2_BOUND)   # unreadable without L2 (review)
    out["N3"] = all(recs[L]["n3_holds"] for L in Ls)            # the SAME test depth_record used for its call basis
    c1, c2 = recs[Ls[-2]]["call"], recs[Ls[-1]]["call"]
    out["two_deepest_calls"] = {str(Ls[-2]): c1, str(Ls[-1]): c2}
    out["N2_comparable"] = c1 == c2 == "comparable"
    out["extrapolation"] = {k: extrapolation(Ls, [recs[L][k]["sd"] for L in Ls]) for k in ("g", "e", "i")}
    deciding = Ls[-2:]
    if any(L not in bterm for L in deciding):                # 1b is part of stage 1: no decision without it
        out.update(outcome="INCOMPLETE", reason=f"stage 1b (the base term) is missing at a deciding depth {deciding}")
        return out
    reads = {L: bterm[L]["read_1b"] for L in deciding}         # read FIRST: either stop overrides the side call
    if "MOOT" in reads.values():                               # a readable depth that says moot is the stronger statement
        out.update(outcome="MOOT", reason=f"|delta0_F|/|delta0_E| > {MOOT_RATIO} at a deciding depth {reads}: stage 2 cannot lift G1")
        return out
    if "INDETERMINATE" in reads.values():
        out.update(outcome="INDETERMINATE", reason=f"|delta0_E| < {GUARD_Z} SE at a deciding depth {reads}: 1b cannot be read there")
        return out
    if c1 != c2:
        out.update(outcome="STOP_DEPTH_DEPENDENT", reason="the side call differs between the two deepest depths: unknown at 48 layers")
    elif c2 == "E":
        out.update(outcome="STAGE2", reason="E-dominant at the two deepest depths: an fp32 evaluator may cut the spread; Unsloth next")
    else:
        out.update(outcome="STOP", reason=f"{c2} at the two deepest depths: no lower-noise reference within reach")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    base = os.path.dirname(os.path.abspath(a.spec))
    for d in spec["depths"]:                        # paths in the spec are relative to it
        for grp in [d] + ([d["unsloth"]] if d.get("unsloth") else []):
            grp["inloop"] = [os.path.join(base, p) for p in grp["inloop"]]
            grp["eval"] = [os.path.join(base, p) for p in grp["eval"]]
            grp["fref"] = os.path.join(base, grp["fref"])
    try:
        rec = reduce(spec)
    except (ValueError, KeyError, OSError) as e:
        print(f"HPN_UNUSABLE {e}")
        return 2
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"HPN={rec['outcome']} depths={rec['fitted_depths']} calls={rec.get('two_deepest_calls')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
