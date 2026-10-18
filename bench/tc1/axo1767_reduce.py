# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""#1767's four-score reading (bench/tc1/AXO1767-PREREG.md): where axolotl's in-loop held-out number and its exported adapter part.

Scores, each the mean held-out loss over the same 8 rows:
  (1) in-loop   the axolotl arm's own eval at its last step (kernels on), from its receipt's eval_curve
  (2) evaluator hp1_eval.py's eager forward on the canonical export, from eval_shake.json
  (3) off       the saved PEFT adapter in a fresh axolotl model with the kernels OFF (PEFT's reference forward), tc1_arm --rescore-adapter
  (4) on        the same saved adapter in a fresh axolotl model with the kernels ON (the training configuration)

Bands, fixed before any data (the registration): two scores MATCH when |delta| <= MATCH_BAND, DIFFER when |delta| >= DIFFER_BAND,
and anything between is INCONCLUSIVE. Instrument checks first; then the 2x2 on A = (4) vs (1) and B = (3) vs (2), with C = (3) vs (4)
splitting the both-match cell. Exit 0 with a VERDICT line for every reading, including VOID and INCOMPLETE ones.
    python3 axo1767_reduce.py --arm AX.json --eval eval_shake.json [--on ON.json --off OFF.json] [--control E4B.json] --out fourscore.json
"""
import argparse
import json
import sys

MATCH_BAND = 0.01      # nats, 8-row mean: twice hp1k-5090-8's largest step-0 disagreement (0.0044), rounded up
DIFFER_BAND = 0.02     # nats: HP1's registered round-trip tolerance (hp1_eval.ROUNDTRIP_TOL)

VERDICTS = {
    ("MATCH", "MATCH", "DIFFER"): ("KERNELS", "the saved adapter reproduces the in-loop number only under axolotl's kernels; PEFT's reference "
                                              "forward and the evaluator agree on it: the kernels compute a different function than PEFT's math"),
    ("MATCH", "MATCH", "MATCH"): ("INCONCLUSIVE", "every pair is inside the match band, yet (2)-(1) differs: the bands cannot place the gap"),
    ("MATCH", "DIFFER", None): ("EXPORT", "the kernels-on reload reproduces the in-loop number, but PEFT's reference forward on the same "
                                          "adapter disagrees with the evaluator: the canonical export or the evaluator reads the adapter differently"),
    ("DIFFER", "MATCH", None): ("PROCESS_STATE", "the saved adapter does not reproduce the in-loop number even under the kernels, while the "
                                                 "reference forward agrees with the evaluator: the training process held state the adapter does not carry"),
    ("DIFFER", "DIFFER", None): ("NEITHER", "neither pair matches: no single-cause reading; the four numbers are reported as they are"),
}


def band(d):
    a = abs(d)
    return "MATCH" if a <= MATCH_BAND else "DIFFER" if a >= DIFFER_BAND else "INCONCLUSIVE"


def _arm_scores(path):
    r = json.load(open(path))
    curve = r.get("eval_curve") or []
    if not curve:
        raise ValueError(f"{path}: no eval_curve")
    return r, curve[0]["heldout_loss"], curve[-1]["heldout_loss"], curve[-1]["step"]


def _eval_arm(ev, tag):
    for a in ev.get("arms", []):
        if a.get("tag") == tag:
            return a
    return None


def _eval_zero(ev, tag):
    for b in ev.get("bases", []):
        z = (b.get("zero_adapter_vs_inloop_step0") or {}).get(tag)
        if z is not None:
            return z
    return None


def reduce(arm, evalj, on=None, off=None, control=None):
    out = {"reducer": "bench/tc1/axo1767_reduce.py", "match_band": MATCH_BAND, "differ_band": DIFFER_BAND, "inputs": {
        "arm": arm, "eval": evalj, "on": on, "off": off, "control": control}, "checks": {}, "scores": {}, "pairs": {}}
    r, s0, s1, step = _arm_scores(arm)
    tag = r.get("tag")
    ev = json.load(open(evalj))
    ea = _eval_arm(ev, tag)
    if ea is None or (ea.get("evaluator") or {}).get("row_mean") is None:
        out.update(verdict="VOID", why=f"the evaluator has no score for {tag}")
        return out
    sc = out["scores"]
    sc["1_inloop"], sc["2_evaluator"] = s1, ea["evaluator"]["row_mean"]
    sc["inloop_step0"], sc["inloop_step"] = s0, step
    z = _eval_zero(ev, tag)
    out["checks"]["evaluator_zero_vs_inloop_step0"] = z
    gap = sc["2_evaluator"] - sc["1_inloop"]
    out["pairs"]["2_minus_1"] = {"delta": round(gap, 6), "band": band(gap)}
    bad = []
    if z is None or band(z) != "MATCH":
        bad.append(f"the evaluator's zero adapter vs the in-loop step 0 is {z} (needs MATCH, |d| <= {MATCH_BAND})")
    if control:
        cr, c0, c1, _ = _arm_scores(control)
        ca = _eval_arm(ev, cr.get("tag"))
        cd = None if ca is None or (ca.get("evaluator") or {}).get("row_mean") is None else ca["evaluator"]["row_mean"] - c1
        out["checks"]["control_round_trip"] = {"tag": cr.get("tag"), "delta": None if cd is None else round(cd, 6),
                                               "band": None if cd is None else band(cd)}
        if cd is None or band(cd) != "MATCH":
            bad.append(f"the control arm {cr.get('tag')}'s round trip is {cd} (needs MATCH)")
    for name, path in (("3_off", off), ("4_on", on)):
        if path:
            q = json.load(open(path))
            if q.get("status") != "ok" or q.get("adapter_loss") is None:
                bad.append(f"{name}: {path} is not an ok rescore ({q.get('status')}: {str(q.get('reason'))[:120]})")
                continue
            sc[name] = q["adapter_loss"]
            zd = q["zero_adapter_loss"] - s0
            out["checks"][f"{name}_zero_vs_inloop_step0"] = {"delta": round(zd, 6), "band": band(zd)}
            if band(zd) != "MATCH":
                bad.append(f"{name}'s zero adapter vs the in-loop step 0 is {round(zd, 6)} (needs MATCH)")
    if bad:
        out.update(verdict="VOID", why="; ".join(bad))
        return out
    if band(gap) != "DIFFER":
        out.update(verdict="NOT_REPRODUCED" if band(gap) == "MATCH" else "INCONCLUSIVE",
                   why=f"the (2)-(1) gap is {round(gap, 6)} ({band(gap)}); #1767's reading needs it at >= {DIFFER_BAND}")
        return out
    if "3_off" not in sc or "4_on" not in sc:
        out.update(verdict="INCOMPLETE", why="the gap reproduces; scores (3) and (4) are missing")
        return out
    A, B, C = sc["4_on"] - sc["1_inloop"], sc["3_off"] - sc["2_evaluator"], sc["3_off"] - sc["4_on"]
    for k, d in (("A_4_minus_1", A), ("B_3_minus_2", B), ("C_3_minus_4", C)):
        out["pairs"][k] = {"delta": round(d, 6), "band": band(d)}
    ba, bb, bc = band(A), band(B), band(C)
    if "INCONCLUSIVE" in (ba, bb) or (ba == bb == "MATCH" and bc == "INCONCLUSIVE"):
        out.update(verdict="INCONCLUSIVE", why=f"A {ba}, B {bb}, C {bc}: a pair sits between the bands")
        return out
    key = (ba, bb, bc if ba == bb == "MATCH" else None)
    out["verdict"], out["why"] = VERDICTS[key]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="the axolotl shake arm's receipt")
    ap.add_argument("--eval", required=True, help="hp1_eval.py's output over the shake arms")
    ap.add_argument("--on", default=None, help="tc1_arm --rescore-adapter with --axolotl-kernels on")
    ap.add_argument("--off", default=None, help="tc1_arm --rescore-adapter with --axolotl-kernels off")
    ap.add_argument("--control", default=None, help="the e4b shake arm's receipt (its round trip must MATCH)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = reduce(a.arm, a.eval, a.on, a.off, a.control)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"AXO1767 VERDICT {out['verdict']}: {out['why']}")
    print("AXO1767 SCORES " + json.dumps(out["scores"]) + " PAIRS " + json.dumps(out["pairs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
