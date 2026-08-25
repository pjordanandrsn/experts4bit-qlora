# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-f1-stageB. Bars fixed by the prereg;
run --self-test before pointing it at receipts."""

import argparse
import json
import sys

PASS_MS = 10.5
PARTIAL_MS = 12.0
AA_MARGIN_MS = 1.48      # half the PASS margin from the 13.46 anchor


def _step(arm):
    v = (arm or {}).get("step_ms_clean") or (arm or {}).get("step_ms")
    if isinstance(v, list):
        v = sorted(v)[len(v) // 2] if v else None
    return v


def _tokens(arm):
    t = (arm or {}).get("tokens")
    if isinstance(t, dict):                       # {row: [ids]}
        t = t.get("0") or (list(t.values())[0] if t else None)
    return list(t) if t else None


def verdict(rep):
    b0a, b0b = rep.get("b0_a"), rep.get("b0_b")
    treat, name = rep.get("treatment"), rep.get("treatment_name", "B1")
    for label, arm in (("b0_a", b0a), ("b0_b", b0b), (name, treat)):
        if _step(arm) is None:
            return ("REFUSE", f"{label}: no step time in the arm")
    # 2. A/A before A/B
    sa, sb = _step(b0a), _step(b0b)
    spread = abs(sa - sb)
    if spread > AA_MARGIN_MS:
        return ("REFUSE", f"A/A spread {spread:.2f} ms > {AA_MARGIN_MS} ms "
                "-- box disqualified before the treatment is read")
    # 1. token identity
    t0, tt = _tokens(b0a), _tokens(treat)
    if not t0 or not tt:
        return ("REFUSE", "missing generated tokens -- a vacuous "
                "identity check cannot certify a fusion")
    n = min(len(t0), len(tt))
    if n == 0 or t0[:n] != tt[:n]:
        first = next((i for i in range(n) if t0[i] != tt[i]), None)
        return ("REFUSE", f"token identity broken at index {first} "
                f"-- a fusion that moves a token is refused, not disclosed")
    # 4. no silent recompile inside the timed window
    rc = (treat or {}).get("recompiles_in_window")
    if rc:
        return ("REFUSE", f"{rc} dynamo recompiles inside the timed "
                "window -- the arm timed compilation, not the kernels")
    st = _step(treat)
    base = min(sa, sb)
    gain = base - st
    if st <= PASS_MS:
        return ("PASS", f"{name} step {st:.2f} ms <= {PASS_MS} "
                f"({1000/st:.1f} tok/s, {gain:.2f} ms removed)")
    if st <= PARTIAL_MS:
        if spread >= gain / 2:
            return ("REFUSE", f"PARTIAL range but A/A spread "
                    f"{spread:.2f} >= half the gain {gain:.2f}")
        return ("PARTIAL", f"{name} step {st:.2f} ms in "
                f"({PASS_MS}, {PARTIAL_MS}] ({1000/st:.1f} tok/s, "
                f"{gain:.2f} ms removed, A/A {spread:.2f})")
    return ("REFUTED", f"{name} step {st:.2f} ms > {PARTIAL_MS} "
            f"({1000/st:.1f} tok/s) -- the block is not addressable "
            "by this mechanism")


_MISSING = object()


def _fab(step, aa=0.2, toks=_MISSING, treat_toks=_MISSING, rc=0):
    """`or`-defaulting would swallow an explicit empty token list and
    silently turn the vacuous-identity case into a passing one, so the
    sentinel is load-bearing rather than style."""
    t = [1, 2, 3, 4] if toks is _MISSING else toks
    tt = t if treat_toks is _MISSING else treat_toks
    return {"b0_a": {"step_ms_clean": 13.46, "tokens": t},
            "b0_b": {"step_ms_clean": 13.46 + aa, "tokens": t},
            "treatment_name": "B1",
            "treatment": {"step_ms_clean": step, "tokens": tt,
                          "recompiles_in_window": rc}}


def self_test():
    cases = [
        (_fab(9.8), "PASS"),
        (_fab(10.5), "PASS"),                    # boundary: <= passes
        (_fab(11.0), "PARTIAL"),
        (_fab(12.0), "PARTIAL"),                 # boundary
        (_fab(12.6), "REFUTED"),
        (_fab(9.8, aa=2.0), "REFUSE"),           # A/A too wide
        (_fab(9.8, treat_toks=[1, 2, 9, 4]), "REFUSE"),   # token moved
        (_fab(9.8, toks=[]), "REFUSE"),          # vacuous identity
        (_fab(9.8, rc=3), "REFUSE"),             # recompiled in window
        # PARTIAL with an A/A spread >= half the gain must refuse, not
        # report a gain the box cannot resolve
        (_fab(11.9, aa=1.0), "REFUSE"),
    ]
    for rep, want in cases:
        got, why = verdict(rep)
        assert got == want, (got, want, why)
    # a PARTIAL whose spread is comfortably under half the gain stands
    got, _ = verdict(_fab(11.0, aa=0.2))
    assert got == "PARTIAL", got
    print(f"self-test PASS ({len(cases)+1} cases: all four verdicts, "
          "both boundaries, and every registered refusal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.report:
        sys.exit("need a report path (or --self-test)")
    v, why = verdict(json.loads(open(a.report).read()))
    print(f"F1 STAGE B VERDICT: {v}\n  {why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
