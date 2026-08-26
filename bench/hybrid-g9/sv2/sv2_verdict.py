# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-sv2 adjudication. The census REFUSEs on its own gates or
stands; it cannot fail a speed bar. Of the two registered routes to
250, only the SPECULATION route is mechanical (cells and thresholds
were fixed in the prereg); the COMPOSITION route needs a human mapping
of kernel slices to treatments, so this calculator emits the slice
table the RESULTS must sum against the 2.48 ms bar rather than
adjudicating it.

Report shape (composed on the box):
  {"on_a": <b1d graph receipt>, "on_b": <b1d graph receipt>,
   "budget": <f1/step_budget.py output for the replay window>,
   "verify": {"8": <s2-verify time receipt>, "16": ..., "32": ...}}

Gates (all REFUSE, in order):
  G1 on_a/on_b step spread <= 2% and token streams identical and
     zero recompiles in either window (per-arm determinism).
  G2 anchor health: median knob-ON step within +/-5% of the CERTIFIED
     6.476 ms point (SV1 K6-B). Outside, the ms-denominated 250 frame
     does not transfer to this box.
  G3 census coverage: the budget's own coverage gate passed AND its
     device us/step reconciles with the measured step
     (0.80 <= device/step <= 1.05) -- below, kineto missed replay
     kernels and the budget cannot anchor the composition frame.
  G4 all three verify cells present with finite verify_graph_ms.

Speculation route (registered): accept(K) = {8: 2.39, 16: 2.948,
32: 3.447} from RESULTS-s1-acceptance; draft cost 0.0 ms (prompt
lookup, the S3 basis); eff(K) = verify_graph_ms(K) / accept(K);
a cell clears iff eff <= 4.00 ms. Route verdict CLEARED iff any cell
clears, else REFUTED.
"""

import argparse
import json
import math
import sys

CERT_KNOB_MS = 6.476          # SV1 K6-B composed knob point
ANCHOR_TOL = 0.05
AA_TOL = 0.02
COVER_LO, COVER_HI = 0.80, 1.05
ACCEPT = {"8": 2.39, "16": 2.948, "32": 3.447}   # RESULTS-s1-acceptance
DRAFT_MS = 0.0                # prompt lookup (S3 basis)
CELL_BAR_MS = 4.00            # 250 tok/s


def adjudicate(rep):
    out = {"gates": {}, "spec_cells": {}, "census": None,
           "spec_route": None, "slices": None}

    for arm in ("on_a", "on_b"):
        if rep.get(arm, {}).get("recompiles_in_window") != 0:
            return _refuse(out, f"G1: {arm} recompiles_in_window "
                                f"{rep.get(arm, {}).get('recompiles_in_window')!r} != 0")
    a = rep["on_a"]["step_ms_clean"]
    b = rep["on_b"]["step_ms_clean"]
    spread = abs(a - b) / min(a, b)
    out["gates"]["aa_spread"] = spread
    if spread > AA_TOL:
        return _refuse(out, f"G1: A/A spread {spread * 100:.2f}% > "
                            f"{AA_TOL * 100:.0f}%")
    if rep["on_a"]["tokens"] != rep["on_b"]["tokens"]:
        return _refuse(out, "G1: on_a/on_b token streams differ "
                            "(same-config graph runs must be "
                            "deterministic)")

    knob = (a + b) / 2
    out["gates"]["knob_ms"] = knob
    drift = abs(knob - CERT_KNOB_MS) / CERT_KNOB_MS
    out["gates"]["anchor_drift"] = drift
    if drift > ANCHOR_TOL:
        return _refuse(out, f"G2: knob point {knob:.3f} ms is "
                            f"{drift * 100:.1f}% from the certified "
                            f"{CERT_KNOB_MS} ms (> {ANCHOR_TOL * 100:.0f}%)")

    bud = rep["budget"]
    if not bud.get("coverage_ok"):
        return _refuse(out, f"G3: budget coverage "
                            f"{bud.get('coverage', 0) * 100:.1f}% failed "
                            f"its own gate")
    # the FOOTER truth, not the row sum: row-limited tables under-sum,
    # and the footer is what the device actually ran (step_budget.py)
    dev_ms = bud["device_us_per_step_truth"] / 1000.0
    recon = dev_ms / knob
    out["gates"]["census_reconcile"] = recon
    if not (COVER_LO <= recon <= COVER_HI):
        return _refuse(out, f"G3: census device {dev_ms:.3f} ms/step is "
                            f"{recon:.2f}x the measured {knob:.3f} ms "
                            f"step (outside [{COVER_LO}, {COVER_HI}]) "
                            f"-- kineto did not see the replay")

    for k in ACCEPT:
        cell = rep.get("verify", {}).get(k)
        if cell is None or not math.isfinite(
                cell.get("verify_graph_ms", float("nan"))):
            return _refuse(out, f"G4: verify cell K={k} missing or "
                                f"non-finite")

    out["census"] = "OK"
    for k, acc in ACCEPT.items():
        v = rep["verify"][k]["verify_graph_ms"]
        eff = (v + DRAFT_MS) / acc
        out["spec_cells"][k] = {"verify_graph_ms": v, "accept": acc,
                                "eff_ms_per_accepted": eff,
                                "clears": eff <= CELL_BAR_MS}
    cleared = [k for k, c in out["spec_cells"].items() if c["clears"]]
    out["spec_route"] = ("CLEARED", cleared) if cleared else ("REFUTED",)

    out["slices"] = bud.get("rows", [])[:40]
    return out


def _refuse(out, why):
    out["census"] = ("REFUSE", why)
    return out


def render(out):
    if isinstance(out["census"], tuple):
        return f"SV2 CENSUS: REFUSE\n  {out['census'][1]}"
    lines = [f"SV2 CENSUS: OK  (knob {out['gates']['knob_ms']:.3f} ms, "
             f"A/A {out['gates']['aa_spread'] * 100:.2f}%, census "
             f"reconciles at {out['gates']['census_reconcile']:.2f}x)"]
    for k in ("8", "16", "32"):
        c = out["spec_cells"][k]
        lines.append(f"  K={k}: verify {c['verify_graph_ms']:.2f} ms / "
                     f"accept {c['accept']} = "
                     f"{c['eff_ms_per_accepted']:.2f} ms per accepted "
                     f"token -> {'CLEARS' if c['clears'] else 'short of'}"
                     f" {CELL_BAR_MS:.2f}")
    if out["spec_route"][0] == "CLEARED":
        lines.append(f"SV2 SPEC ROUTE: CLEARED at K="
                     f"{','.join(out['spec_route'][1])} -- S4 registers")
    else:
        lines.append("SV2 SPEC ROUTE: REFUTED (no cell clears; the S3 "
                     "negative stands on the current stack)")
    lines.append("SV2 COMPOSITION ROUTE: adjudicated in RESULTS -- sum "
                 "the addressable slices below against the 2.48 ms bar")
    for r in out["slices"][:25]:
        lines.append(f"    {r['us_per_step']:9.1f} us/step  "
                     f"{r['kind']:<12} {r['name'][:70]}")
    return "\n".join(lines)


def _load_step_budget():
    import importlib.util
    import pathlib
    p = (pathlib.Path(__file__).resolve().parent.parent / "f1"
         / "step_budget.py")
    spec = importlib.util.spec_from_file_location("step_budget", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_budget(step_ms, dev_frac=0.97, cover=True):
    """Build the budget dict by running the REAL producer on a
    synthetic kineto table -- an invented fixture dict here let a
    schema mismatch through the self-test once (Bugbot, e4b#276)."""
    import tempfile
    sb = _load_step_budget()
    steps = 8
    dev_us = step_ms * 1000 * dev_frac * steps
    row = (f"  gemv_nf4_grouped_kernel  10.0%  1.000ms  10.0%  "
           f"1.000ms  1.000ms  {dev_us:.3f}us  90.0%  {dev_us:.3f}us  "
           f"1.000us  {steps * 48}")
    # footer above the row sum -> coverage below the producer's own
    # gate; equal -> coverage 1.0
    total = dev_us if cover else dev_us * 2
    txt = (f"profiled replay steps: {steps} (active window: "
           f"{steps}/{steps})\n{row}\n"
           f"Self CUDA time total: {total / 1000:.3f}ms\n")
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                     delete=False) as f:
        f.write(txt)
        path = f.name
    return sb.budget(*sb.parse(path))


def _mk(a=6.47, b=6.48, dev_frac=0.97, v8=25.0, v16=30.0, v32=35.0,
        rec=0, toks_b=None, cover=True):
    toks = list(range(50))
    knob = (a + b) / 2
    return {"on_a": {"step_ms_clean": a, "tokens": toks,
                     "recompiles_in_window": rec},
            "on_b": {"step_ms_clean": b,
                     "tokens": toks_b if toks_b is not None else toks,
                     "recompiles_in_window": 0},
            "budget": _mk_budget(knob, dev_frac, cover),
            "verify": {"8": {"verify_graph_ms": v8},
                       "16": {"verify_graph_ms": v16},
                       "32": {"verify_graph_ms": v32}}}


def self_test():
    r = adjudicate(_mk())
    assert r["census"] == "OK", r
    assert r["spec_route"][0] == "REFUTED", r
    # one clearing cell flips the route: 9.4 / 2.39 = 3.93 <= 4.00
    r = adjudicate(_mk(v8=9.4))
    assert r["spec_route"] == ("CLEARED", ["8"]), r
    # boundary: exactly 4.00 clears; a hair over does not
    r = adjudicate(_mk(v16=2.948 * 4.00))
    assert r["spec_cells"]["16"]["clears"], r
    r = adjudicate(_mk(v16=2.948 * 4.001))
    assert not r["spec_cells"]["16"]["clears"], r
    # refusal directions, one per gate
    for bad, why in ((_mk(b=6.68), "G1"),            # 3.2% spread
                     (_mk(rec=1), "G1"),
                     (_mk(toks_b=[1]), "G1"),
                     (_mk(a=6.9, b=6.92), "G2"),     # 5.3% off cert
                     (_mk(cover=False), "G3"),
                     (_mk(dev_frac=0.5), "G3"),
                     (_mk(dev_frac=1.2), "G3"),
                     (_mk(v32=float("nan")), "G4")):
        r = adjudicate(bad)
        assert isinstance(r["census"], tuple) and \
            r["census"][1].startswith(why), (why, r["census"])
    missing = _mk()
    del missing["verify"]["16"]
    r = adjudicate(missing)
    assert isinstance(r["census"], tuple) and \
        r["census"][1].startswith("G4"), r
    print("sv2_verdict self-test OK (route flip, 4.00 boundary, and "
          "eight refusal directions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.report:
        ap.error("report path or --self-test")
    out = adjudicate(json.load(open(a.report)))
    print(render(out))
    if isinstance(out["census"], tuple):
        sys.exit(3)


if __name__ == "__main__":
    main()
