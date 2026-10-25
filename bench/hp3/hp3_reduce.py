#!/usr/bin/env python3
"""bench/hp3/hp3_reduce.py -- HP3's registered reducer (bench/hp3/HP3-PREREG.md): G1' and the pair verdicts.

Scope label, carried in every verdict line: parity of the exported adapters under one evaluator on byte-identical bases.

`g1`: from the one evaluator's calibration pass (each framework's one-step calibration arm on its own exported base, on R rows),
G1' per framework f against e4b's base, deterministic, with no interval: f's dequantized base digest equals e4b's AND the
evaluator's zero-adapter per-row losses on f's base equal e4b's to ROW_TOL on every row (so the row means are equal). A mismatch
CLOSEs f's pairs ("bases differ: outside this registration"). HP2's delta0 per pair (on identical bases, the in-loop term: the
difference of the two frameworks' training forwards) and each framework's b_f are REPORTED, never gated, with the home-field
direction. Writes --out BEFORE returning; exit 0 when any pair can still be read, 4 on a whole-box CLOSE (a reading, not a
strike; the box tears down), 2 unusable input.

`verdict`: from the confirmation pass (seeds 3410-3412 per framework that passed G1') and the G1' record. Per run, 6(b) VOID iff
|adapter_roundtrip| > ROUNDTRIP_MATCH; an axolotl run also VOIDs unless its arm receipt shows kernels OFF (AXO1767 Amendment 1's
engagement check). Per pair: the seed-paired Delta, its 90 % t-interval and G2' (HP2's realized-margin TOST on Delta alone);
deltaN is reported with the in-loop term named. Then the three-way read. Exit 0 when every pair is read, 3 otherwise.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hp1"))   # in the repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tc1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))                 # on the box, the staged files sit side by side
import hp1_reduce as H1  # noqa: E402  (pooled_sd, order_problems, manifest_problems, t_ppf, _mean, _ss: tested in HP1)
import axo1767_reduce as AX  # noqa: E402  (engagement: AXO1767 Amendment 1's check, tested there)

SCOPE = "parity of the exported adapters under one evaluator on byte-identical bases"
LABELS = ("(i) Unsloth 2026.10.7 + zoo 2026.10.7 is a NEW version, not HP1/HP2's 2026.9.14; "
          "(ii) axolotl 0.20.0 runs kernels OFF (eager experts, no kernels plugin, no lora_*_kernel), not AXO1767's configuration; "
          "(iii) the build is torch 2.14.0+cu130 with bnb 0.50.2: no HP3 number is compared with an HP2 or HPN number")
ROW_TOL = 1e-5           # HPN's N0 band: the per-row losses are recorded to 5 decimals
ROUNDTRIP_MATCH = 0.01   # 6(b), zero-centred, as HP2: the house's round-trip MATCH band (AXO1767-PREREG: MATCH <= 0.01)
CAL_ROWS = 300           # held-out pool rows 97-396, as HP2
FRAMEWORKS = ("e4b", "unsloth", "axolotl")
REFERENCE = "e4b"
PAIRS = (("e4b", "unsloth"), ("e4b", "axolotl"), ("unsloth", "axolotl"))
CAL_TAGS = {"e4b": "hp3cal_e4b_k00_s3407", "unsloth": "hp3cal_unsloth_s3407", "axolotl": "hp3cal_axolotl_s3407"}
MODE = "p"
# HP3 Amendment 2, the (R) fallback (registered before any data; triggered by hp3-shake-2's G1' at L2): e4b vs Unsloth 2026.9.14 on
# HPN's build, one pair, no axolotl. Label (i) is dropped and (iii) names torch 2.8.0+cu128.
LABELS_R = ("(R) the registered fallback: e4b K00 vs Unsloth 2026.9.14 (HP1/HP2's version) on torch 2.8.0+cu128 (HPN's build); "
            "axolotl waits for an amendment; "
            "(iii) the build is torch 2.8.0+cu128 with Unsloth's own bitsandbytes resolve: no HP3 (R) number is compared with an HP2 or HPN number")


_P_GLOBALS = (FRAMEWORKS, PAIRS, LABELS)


def set_mode_p():
    """Back to build (P): three frameworks, three pairs, (P)'s labels (tests switch modes in one process)."""
    global FRAMEWORKS, PAIRS, LABELS, MODE
    (FRAMEWORKS, PAIRS, LABELS), MODE = _P_GLOBALS, "p"


def set_fallback_r():
    """Switch this module to (R): two frameworks, one pair, the (R) labels; the three-way read does not apply."""
    global FRAMEWORKS, PAIRS, LABELS, MODE
    FRAMEWORKS, PAIRS, LABELS, MODE = ("e4b", "unsloth"), (("e4b", "unsloth"),), LABELS_R, "r"
RUN_PREFIX = {"e4b": "hp3_k00_s", "unsloth": "hp3_unsloth_s", "axolotl": "hp3_axolotl_s"}
SEEDS = (3410, 3411, 3412, 3413, 3414)   # Amendment 1: five confirmation seeds per framework (was HP1's 3410-3412)
EXIT_OK, EXIT_UNUSABLE, EXIT_INCOMPLETE, EXIT_G1_CLOSE = 0, 2, 3, 4

WORDING = {
    "BEATS": "{a}'s held-out loss is lower than {b}'s: the 90 % interval excludes 0 and |mean| > sigma_seed",
    "BEHIND": "{a}'s held-out loss is higher than {b}'s: the 90 % interval excludes 0 and |mean| > sigma_seed",
    "PARITY": "no measurable difference at margin m: the 90 % interval lies inside +-m",
    "INCONCLUSIVE": "no measurable effect, with its interval: neither inside +-m nor clear of 0",
    "CLOSE": "bases differ: outside this registration",
    "NOT_READ": "not read: {why}",
}


def _base(ev, tag):
    recs = [b for b in ev.get("bases", []) if tag in (b.get("zero_adapter_row_gaps") or {})]
    if len(recs) > 1:
        raise ValueError(f"expected at most one base with per-row zero-adapter gaps for {tag!r}, found {len(recs)}")
    return recs[0] if recs else None


def g1(ev, rows=CAL_ROWS):
    """The G1' record from an hp1_eval output dict. Raises ValueError on input that cannot be read as HP3's calibration."""
    bases, gaps, per = {}, {}, {}
    for fw in FRAMEWORKS:
        b = _base(ev, CAL_TAGS[fw])
        if b is None:
            continue
        if not b.get("digest_matches_training_run"):
            raise ValueError(f"{fw}'s calibration base digest does not match its own training run")
        losses = (b.get("zero_adapter") or {}).get("row_losses") or []
        g = b["zero_adapter_row_gaps"][CAL_TAGS[fw]]
        for name, xs in (("zero-adapter row losses", losses), ("per-row gaps", g)):
            if len(xs) != rows or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs):
                raise ValueError(f"{fw}: expected {rows} finite {name}, got {len(xs)}")
        bases[fw], gaps[fw] = b, g
        per[fw] = {"b": round(statistics.mean(g), 6), "sd": round(statistics.stdev(g), 6),
                   "mean_abs_gap": round(statistics.mean(abs(x) for x in g), 6), "max_abs_gap": round(max(abs(x) for x in g), 6),
                   "digest": b.get("digest"), "zero_row_mean": (b.get("zero_adapter") or {}).get("row_mean")}
    if REFERENCE not in bases:
        raise ValueError(f"no calibration base for the reference framework {REFERENCE!r}")
    ref = bases[REFERENCE]
    ref_rows = ref["zero_adapter"]["row_losses"]
    status = {REFERENCE: {"g1": "PASS", "reason": "the reference base"}}
    for fw in FRAMEWORKS:
        if fw == REFERENCE:
            continue
        if fw not in bases:
            status[fw] = {"g1": "ABSENT", "reason": "no calibration base in the evaluator output"}
            continue
        b = bases[fw]
        same_digest = b.get("digest") == ref.get("digest")
        worst = max(abs(x - y) for x, y in zip(b["zero_adapter"]["row_losses"], ref_rows))
        same_mean = b["zero_adapter"].get("row_mean") == ref["zero_adapter"].get("row_mean")
        rec = {"same_digest": same_digest, "max_row_diff": round(worst, 7), "rows_equal": worst <= ROW_TOL, "same_row_mean": same_mean}
        if same_digest and worst <= ROW_TOL and same_mean:
            rec.update(g1="PASS", reason=f"digest equal to {REFERENCE}'s; every zero-adapter row within {ROW_TOL}; row means equal")
        else:
            why = [w for w, bad in (("the dequantized digests differ", not same_digest),
                                    (f"a zero-adapter row differs by {worst:.7f} > {ROW_TOL}", worst > ROW_TOL),
                                    ("the zero-adapter row means differ", not same_mean)) if bad]
            rec.update(g1="CLOSE", reason=f"{WORDING['CLOSE']}: " + "; ".join(why))
        status[fw] = rec
    # REPORTED, never gated: HP2's delta0 per pair. On identical bases it is the in-loop term, -mean_r (I_a(r) - I_b(r)).
    delta0 = {}
    for a, b in PAIRS:
        if a in gaps and b in gaps:
            d = [x - y for x, y in zip(gaps[a], gaps[b])]
            delta0[f"{a}-{b}"] = {"delta0": round(statistics.mean(d), 6), "se": round(statistics.stdev(d) / math.sqrt(rows), 6),
                                  "term": "in-loop (the two frameworks' training forwards at step 0)" if status.get(a, {}).get("g1") == "PASS"
                                  and status.get(b, {}).get("g1") == "PASS" else "base and in-loop (the bases differ)"}
    home = min(per, key=lambda f: abs(per[f]["b"]))
    passing = [f for f in FRAMEWORKS if status.get(f, {}).get("g1") == "PASS"]
    out = {"reducer": "bench/hp3/hp3_reduce.py g1", "scope": SCOPE, "labels": LABELS, "rows": rows, "row_tol": ROW_TOL, "reference": REFERENCE,
           "frameworks": status, "per_framework": per, "delta0_reported": delta0,
           "home_field": {"closest": home, "basis": "smallest |b_f|: the framework whose training forward the evaluator sits closest to"},
           "roundtrip_zero_centred_tol": ROUNDTRIP_MATCH, "readable_pairs": [f"{a}-{b}" for a, b in PAIRS if a in passing and b in passing]}
    out["verdict"] = "PASS" if len(passing) == len(FRAMEWORKS) else "PARTIAL" if out["readable_pairs"] else "CLOSE"
    return out


def _runs(evs):
    """{(framework, seed): arm} for HP3's confirmation tags; anything else is ignored."""
    out = {}
    for ev in evs:
        for a in ev.get("arms", []):
            tag = a.get("tag") or ""
            for fw, prefix in RUN_PREFIX.items():
                if tag.startswith(prefix) and tag[len(prefix):].isdigit():
                    k = (fw, int(tag[len(prefix):]))
                    if k in out:
                        raise ValueError(f"duplicate run {k}")
                    out[k] = a
    return out


def _void(fw, a, receipts):
    """Why run a is VOID, or None. 6(b) on the adapter round trip; for axolotl also the kernels-off engagement check."""
    if a is None:
        return "absent"
    if a.get("status") != "SCORED" or (a.get("evaluator") or {}).get("row_mean") is None:
        return f"status {a.get('status')}"
    rt = a.get("adapter_roundtrip")
    if rt is None:
        return "no zero-centred round trip recorded"
    if abs(rt) > ROUNDTRIP_MATCH:
        return f"6(b): |adapter_roundtrip| {abs(rt):.6f} > {ROUNDTRIP_MATCH}"
    if a.get("evaluator_minus_inloop_final") is None:
        return "no evaluator_minus_inloop_final recorded (the in-loop step-N number is missing)"
    if fw == "axolotl":
        q = receipts.get(a["tag"])
        if q is None:
            return "no arm receipt: the kernels-off engagement cannot be checked"
        bad = AX.engagement("3_off", q)
        if bad:
            return "engagement: " + "; ".join(bad)
    return None


def _order(ranks):
    """True iff the pair verdicts (winner, loser) admit one total order of the frameworks."""
    import itertools
    return any(all(p.index(w) < p.index(lo) for w, lo in ranks) for p in itertools.permutations(FRAMEWORKS))


def paired_reading(name, a_ys, a_recs, b_ys, b_recs, sigma, margin, seeds=SEEDS):
    """hp1_reduce.paired_reading's rule, unchanged, over HP3's seeds (HP1's reads HP1_CONFIRMATION_SEEDS, three): Delta_s = y(a, s) -
    y(b, s); CI90 by t(df n-1); BEATS / BEHIND (interval clear of 0 and |mean| > sigma) before PARITY (TOST within +-margin), else
    INCONCLUSIVE. tests/test_hp3.py checks it equals hp1_reduce.paired_reading at HP1's three seeds."""
    seeds = list(seeds)
    deltas = [a_ys[s] - b_ys[s] for s in seeds]
    k = len(deltas)
    mean = H1._mean(deltas)
    sd = math.sqrt(H1._ss(deltas) / (k - 1))
    se = sd / math.sqrt(k)
    tq = H1.t_ppf(H1.CI_TAIL, k - 1)
    lo, hi = mean - tq * se, mean + tq * se
    if hi < 0 and abs(mean) > sigma:
        verdict = "BEATS"
    elif lo > 0 and abs(mean) > sigma:
        verdict = "BEHIND"
    elif lo >= -margin and hi <= margin:
        verdict = "PARITY"
    else:
        verdict = "INCONCLUSIVE"
    obo = sorted(r["tag"] for r in a_recs + b_recs if r["own_base_only"])
    return {"reading": name, "seeds": seeds, "deltas": dict(zip(map(str, seeds), deltas)), "mean": mean, "sd": sd, "se": se,
            "df": k - 1, "t_crit": tq, "ci90": [lo, hi], "m": margin, "sigma_seed": sigma, "verdict": verdict,
            "any_own_base_only": bool(obo), "own_base_only": obo}


def seed_rho(a_ys, b_ys, seeds=SEEDS):
    """Amendment 1, REPORTED only: the correlation over seeds between the two frameworks' row means (the seed pairing the paired
    difference relies on). None when either side has no spread."""
    x, y = [a_ys[s] for s in seeds], [b_ys[s] for s in seeds]
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx, syy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    if sxx == 0 or syy == 0:
        return None
    return round(sum((p - mx) * (q - my) for p, q in zip(x, y)) / math.sqrt(sxx * syy), 4)


PLAN_COUNTS = (3, 4, 5)   # Amendment 1 part 7: the seed counts a plan may name (0 = host-limited, nothing read)


def plan_seeds(plan):
    """(n, seeds, reason) from the runner's hp3/seed_plan.json (Amendment 1 part 7), or ValueError. The plan is decided by timing alone,
    so it must say so; it names a PREFIX of SEEDS of length 3, 4 or 5, or 0 (host-limited)."""
    if plan is None:
        return len(SEEDS), list(SEEDS), "no seed plan: the registered five seeds"
    n = plan.get("n")
    if type(n) is not int or (n != 0 and n not in PLAN_COUNTS):
        raise ValueError(f"seed plan n={n!r} is not 0, 3, 4 or 5")
    if plan.get("basis") != "timing":
        raise ValueError(f"seed plan basis {plan.get('basis')!r} is not 'timing'")
    if plan.get("seeds") != list(SEEDS[:n]):
        raise ValueError(f"seed plan seeds {plan.get('seeds')!r} are not the first {n} of {list(SEEDS)}")
    return n, list(SEEDS[:n]), str(plan.get("reason") or "")


def verdict(evs, g1rec, receipts, plan=None):
    """The verdict record from the confirmation evaluator output(s), the G1' record, the arm receipts {tag: receipt} and the seed plan."""
    n_seeds, seeds, plan_reason = plan_seeds(plan)
    arms = _runs(evs)
    ys, recs, voids, rho = {f: {} for f in FRAMEWORKS}, {f: [] for f in FRAMEWORKS}, {}, {f: {} for f in FRAMEWORKS}
    ok_fw = {f for f, st in (g1rec.get("frameworks") or {}).items() if st.get("g1") == "PASS"}
    for fw in sorted(ok_fw):
        for s in seeds:
            a = arms.get((fw, s))
            why = _void(fw, a, receipts)
            if why:
                voids[f"{fw} seed {s}"] = why
                continue
            ys[fw][s] = a["evaluator"]["row_mean"]
            rho[fw][s] = a.get("evaluator_minus_inloop_final")
            recs[fw].append({"tag": a["tag"], "seed": s, "order": a.get("order"), "own_base_only": bool(a.get("own_base_only"))})
    out = {"reducer": "bench/hp3/hp3_reduce.py verdict", "scope": SCOPE, "labels": LABELS, "g1": g1rec.get("verdict"), "void": voids,
           "y": {fw: {str(s): v for s, v in d.items()} for fw, d in ys.items() if d}, "pairs": {},
           "seed_count": n_seeds, "seeds_read": seeds, "seed_plan_reason": plan_reason,
           "reduced_power": (f"a reduced-power read on {n_seeds} seeds (Amendment 1 part 7), never upgraded later" if 0 < n_seeds < len(SEEDS)
                             else None)}
    for a, b in PAIRS:
        name = f"{a}-{b}"
        if n_seeds == 0:
            out["pairs"][name] = {"verdict": "NOT_READ", "scope": SCOPE,
                                  "wording": WORDING["NOT_READ"].format(why=f"host-limited, no seed read ({plan_reason})")}
            continue
        if a not in ok_fw or b not in ok_fw:
            closed = [f for f in (a, b) if f not in ok_fw]
            st = [(g1rec.get("frameworks") or {}).get(f, {}).get("g1", "ABSENT") for f in closed]
            v = "CLOSE" if "CLOSE" in st else "NOT_READ"
            out["pairs"][name] = {"verdict": v, "wording": WORDING["CLOSE"] if v == "CLOSE"
                                  else WORDING["NOT_READ"].format(why=f"{', '.join(closed)} has no calibration base"), "scope": SCOPE}
            continue
        bad = [k for k in voids if k.split(" ")[0] in (a, b)]
        if bad:
            out["pairs"][name] = {"verdict": "NOT_READ", "wording": WORDING["NOT_READ"].format(why="VOID runs: " + ", ".join(sorted(bad))),
                                  "scope": SCOPE}
            continue
        probs = H1.order_problems(recs[a] + recs[b])
        if probs:
            out["pairs"][name] = {"verdict": "NOT_READ", "order_problems": probs, "scope": SCOPE,
                                  "wording": WORDING["NOT_READ"].format(why="the runs do not share one data order per seed")}
            continue
        sigma, sigma_df, _ = H1.pooled_sd([(f"P:{a}", list(ys[a].values())), (f"P:{b}", list(ys[b].values()))])
        m = min(max(sigma, H1.MARGIN_FLOOR), H1.MARGIN_CAP)
        head = paired_reading(f"{a} vs {b}", ys[a], recs[a], ys[b], recs[b], sigma, m, seeds=seeds)
        e = [rho[a][s] - rho[b][s] for s in seeds]
        out["pairs"][name] = {"verdict": head["verdict"], "wording": WORDING[head["verdict"]].format(a=a, b=b), "scope": SCOPE,
                              "m": m, "sigma_seed": sigma, "sigma_seed_df": sigma_df, "headline": head,
                              "rho_seed_reported": seed_rho(ys[a], ys[b], seeds=seeds),
                              "deltaN_reported": {"mean": round(statistics.mean(e), 6), "per_seed": e,
                                                  "term": "in-loop (the two frameworks' training forwards at step N)"}}
    vs = {k: p["verdict"] for k, p in out["pairs"].items()}
    if MODE == "r":                                  # (R): one pair, so the pair's verdict is the reading; no three-way read
        read = all(v not in ("CLOSE", "NOT_READ") for v in vs.values())
        out["three_way"] = {"verdict": "NOT_APPLICABLE", "why": "(R) reads one pair (e4b-unsloth); axolotl waits for an amendment",
                            "pairs_read": sum(v not in ("CLOSE", "NOT_READ") for v in vs.values())}
        out["home_field"] = g1rec.get("home_field")
        return out, (EXIT_OK if read else EXIT_INCOMPLETE)
    if any(v in ("CLOSE", "NOT_READ") for v in vs.values()):
        three = "NOT_READ"
    elif all(v == "PARITY" for v in vs.values()):
        three = "THREE_WAY_PARITY"
    elif all(v in ("BEATS", "BEHIND") for v in vs.values()):
        ranks = [(k.split("-")[0], k.split("-")[1]) if v == "BEATS" else (k.split("-")[1], k.split("-")[0]) for k, v in vs.items()]
        three = "ORDERED" if _order(ranks) else "MIXED"
    else:
        three = "MIXED"
    per = g1rec.get("per_framework") or {}
    closed = {k: {"digests": {f: (per.get(f) or {}).get("digest") for f in k.split("-")}, "wording": p["wording"]}
              for k, p in out["pairs"].items() if p["verdict"] == "CLOSE"}
    unread = {k: p["wording"] for k, p in out["pairs"].items() if p["verdict"] == "NOT_READ"}
    out["three_way"] = {"verdict": three, "pairs_read": sum(v not in ("CLOSE", "NOT_READ") for v in vs.values()),
                        "closed_pairs": closed, "unread_pairs": unread,
                        "correction": "none: each pair is its own registered question; THREE_WAY_PARITY is an intersection of TOSTs"}
    out["home_field"] = g1rec.get("home_field")
    return out, (EXIT_OK if three != "NOT_READ" else EXIT_INCOMPLETE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("g1")
    a1.add_argument("eval_json")
    a1.add_argument("--rows", type=int, default=CAL_ROWS)
    a1.add_argument("--fallback-r", action="store_true", help="HP3 Amendment 2's (R): e4b vs Unsloth only")
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("verdict")
    a2.add_argument("evals", nargs="+")
    a2.add_argument("--g1", required=True)
    a2.add_argument("--arm-receipt", action="append", default=[], help="an axolotl confirmation arm's receipt (its tag inside)")
    a2.add_argument("--manifest", action="append", default=[])
    a2.add_argument("--seed-plan", help="the runner's hp3/seed_plan.json (Amendment 1 part 7); every evaluator output must carry its sha256")
    a2.add_argument("--fallback-r", action="store_true", help="HP3 Amendment 2's (R): e4b vs Unsloth only")
    a2.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.fallback_r:
        set_fallback_r()
    try:
        if a.cmd == "g1":
            rec = g1(json.load(open(a.eval_json)), a.rows)
            json.dump(rec, open(a.out, "w"), indent=1)                    # the record is written before the gate returns
            print(f"HP3_G1={rec['verdict']} readable={','.join(rec['readable_pairs']) or 'none'} "
                  f"frameworks={json.dumps({f: s['g1'] for f, s in rec['frameworks'].items()}, sort_keys=True)}")
            return EXIT_G1_CLOSE if rec["verdict"] == "CLOSE" else EXIT_OK
        mprob = H1.manifest_problems(a.manifest, a.evals) if a.manifest else []
        if mprob:
            json.dump({"reducer": "bench/hp3/hp3_reduce.py verdict", "verdict": "VOID", "manifest_problems": mprob}, open(a.out, "w"), indent=1)
            print(f"HP3_VERDICT=VOID manifest: {mprob[:3]}")
            return EXIT_INCOMPLETE
        receipts = {}
        for p in a.arm_receipt:
            q = json.load(open(p))
            receipts[q.get("tag")] = q
        evs = [json.load(open(p)) for p in a.evals]
        plan = None
        if a.seed_plan:                       # Amendment 1 part 7: the content binding (no mtime is trusted)
            raw = open(a.seed_plan, "rb").read()
            want = hashlib.sha256(raw).hexdigest()
            bad = [p for p, ev in zip(a.evals, evs) if ev.get("plan_sha256") != want]
            if bad:
                raise ValueError(f"the evaluator output(s) {bad} do not carry this seed plan's sha256 {want[:12]} (plan_sha256 missing, null or "
                                 "different): the plan was not fixed before the evaluator's pass, or another plan was")
            plan = json.loads(raw)
        else:
            bound = [p for p, ev in zip(a.evals, evs) if ev.get("plan_sha256") is not None]
            if bound:
                raise ValueError(f"the evaluator output(s) {bound} carry a plan_sha256 but no --seed-plan was given: the runner's plan is missing")
        rec, code = verdict(evs, json.load(open(a.g1)), receipts, plan)
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"HP3_VERDICT three_way={rec['three_way']['verdict']} "
              + " ".join(f"{k}={p['verdict']}" for k, p in rec["pairs"].items()) + f" ({SCOPE})")
        return code
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        print(f"HP3 UNUSABLE {type(e).__name__}: {e}")
        return EXIT_UNUSABLE


if __name__ == "__main__":
    sys.exit(main())
