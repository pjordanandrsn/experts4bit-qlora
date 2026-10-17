# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's milestone verdict, applied mechanically from the registered decision rules (bench/q235/PREREG-q235.md, "Decision rules for
the eventual box" as Amendments 2 and 3 amend them), never read by eye.

Inputs: the box's fetched run directory (``q235_run.sh``'s layout: ``receipts/run.json``, ``receipts/card.txt``, ``receipts/box/``,
``receipts/repeat/`` with the in-process ``plant_routing/`` and ``plant_epilogue/``, ``receipts/tokens.pt``,
``TC1_EXIT_CODE.<nonce>``, ``REFUSAL``, ``summary.txt``) and the full-depth layer-streamed bf16 reference made off the box
(``layer_stream_ref.py``, the same tokens).

**VOID** (no reading) when the run is not a science run, or it ended in a refusal (a named host floor: exit 13, 14, 17 or 18 with its
``REFUSAL``), a host fault (``host-fault.json``), a time-left guard (the deadline reserve) or a harness error (any other exit but the
STRICT failure's 12, or a record missing); and on Amendment 3's instrument checks, which come before any band:
- the null (the repeat's clean step-0 forward) must be the box: every layer within ``NULL_FLOOR`` relative of the box's step-0 output;
- each plant must engage, and only at its layer: the layers before it within ``NULL_FLOOR`` of the null's (bit-for-bit equality
  is reported), the planted layer differing from it;
- each plant must fail the band: its contribution error at the planted layer above ``BAND``.

**MILESTONE PASS** iff all of these hold, else **FAIL** naming each rule that fails:
1. the step completed (exit 0, or 12 when a STRICT rule failed after both steps) on the registered class (``receipts/card.txt``);
2. every step's loss is finite;
3-5. Amendment 2's STRICT rules and the frozen base, re-derived from ``box.json`` (``q235_box.strict_failures``, fail-closed);
6. the fresh-process repeat of step 0 within the larger of the in-process spread and 1e-4 relative (a miss is a finding with its size);
7. Amendment 3's band: the box's contribution error u_i (``q235_compare``'s ``update_rel_err``) at most ``BAND`` on every layer of
   the interior class, ceil(4n / 48) to floor(44n / 48).
Exit 12 stops the run after the box, so a STRICT failure carries no repeat and no plants: it is a FAIL on rules 3-5 with 6 and 7 not read.

**Shape B** (Amendment 6; a box record with ``shape_b``, from ``q235_box.py --micro-batch``) adds:
8. at each step's parameters, rule 8's check passes under deterministic algorithms (``rule8``): the device accumulation (four
   backwards into ``p.grad``) against the same-shapes reference, the host fp32 sum of the same four micro-batches' gradients taken
   one at a time: the per-tensor relative difference at most ``GA_BAR`` on every compared tensor, and no tensor exactly zero in the
   reference but not in the accumulation.
It is VOID while ``GA_BAR`` is unregistered, and on its own instrument checks: the check passes must have run deterministic with the
default restored after them; the floor (two deterministic accumulations against each other) must sit at most ``GA_BAR / 3``; and each
REGISTERED planted accumulation bug (``GA_PLANTS``; a mean of micro-batch means, an overwrite that keeps only the last micro-batch)
must read above ``GA_BAR`` against the same reference. Reported with it: the global cosine and the norm ratio, the registered step's
own default-mode accumulation against the reference (``ga_mb_check``), shape A against shape B in every view (``ga_check``), and the
router top-k flips between them. Shape A's records carry no ``shape_b``, so rule 8 does not apply to them and their verdict is unchanged.

**Amendment 8** (a box record with ``amendment8``; every box from ``q235_run.sh`` at or after it) adds:
9. every decoder layer's output gradient (``dx``, dL/dh_i, i = 0..n-1) recorded, finite and nonzero at both steps, in every
   forward-backward of the step (one per micro-batch under shape B). It is STRICT as well, so a failure also fails rules 3-5;
10. the held-out loss on the registered held-out rows (``heldout``: before the update, after step 0, after step 1) finite at
    every read, and no post-update read above ``before x (1 + HELDOUT_BAND)`` (one-sided).
Rule 10 is a ONE-SIDED blow-up bound ("did the update make held-out text worse?"): every read finite, and each post-update read
at most ``before x (1 + HELDOUT_BAND)``, ``HELDOUT_BAND`` = 0.10; no lower bound, since two steps that teach the shared template
lower held-out loss as they lower the step's own (the 30B rehearsal: -38.6 %). The step's own rows, read the same way, ride along
as reporting. Rule 10 is VOID if the band is unset, and an Amendment 8 box without its held-out read is a harness error.
Records without ``amendment8`` (runs 2 and 3) carry neither field: rules 9 and 10 do not apply and their verdicts are unchanged.

**Reported, not gating:** the box's full u profile (the layers outside the interior class without a bar), the full-state profile with
its median (the upper middle) and maximum, the router overlap and set agreement, the loss gap, each plant's u after its layer, the
device peak against the registered estimate (a deviation above 15 % names an estimator issue), the bytes moved per step against the
estimate's link line, and the steps' elapsed time.

Exit codes: PASS 0, FAIL 1, VOID 2. ``verdict.json`` holds every number the verdict used.

    python q235_verdict.py RUN_DIR --ref REF_DIR --out VERDICT.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys

import torch

CARD_RE = r"^NVIDIA RTX PRO 6000"              # the registered class (q235_run.sh's science CARD_RE)
ESTIMATE_DEVICE_GIB = 24.76                    # e4b's estimate for the registered shape (bf16 adapters, 8 x 331; gate2/capacity-2648.json)
ESTIMATE_LINK_GIB = 237.94                     # the estimate's link line per micro-batch (PREREG "Refusal points" 6)
EXIT = {"PASS": 0, "FAIL": 1, "VOID": 2}
REFUSAL_CODES = {"13", "14", "17", "18"}
BAND = 0.775                                   # Amendment 3: 2 x the 30B box's largest interior contribution error (0.3876)
NULL_FLOOR = 1e-3                              # Amendment 3: the null against the box, and a plant's earlier layers against the null
GA_BAR = 0.0231                                # Amendment 6: 3 x the 30B's s(48) = 3 x 0.00770, third calibration (am6/)
GA_PLANTS = ("mean", "overwrite")              # Amendment 6: the plants the calibration registers (a plant short of 3 x BAR is dropped)
PLANTS = ("routing", "epilogue")
HELDOUT_BAND = 0.10                            # Amendment 8: the largest held-out INCREASE allowed after an update (one-sided)


def l2_fields(box):
    """Amendment 8 (a box record with ``amendment8``): rule 9, every decoder layer's output gradient recorded, finite and
    nonzero at both steps (also STRICT, in rules 3-5); rule 10, a one-sided blow-up bound: every held-out read finite, and no
    post-update read (after step 0, after step 1) above ``before x (1 + HELDOUT_BAND)``. A drop is learning, never a failure.
    Returns ``(rules, the VOID's reason or None)``."""
    a8 = box.get("amendment8") or {}
    try:
        dx = {k: box[k]["dx"] for k in ("step", "step1")}
        r9 = {"ok": all(v["finite_nonzero"] is True for v in dx.values()), "layers": dx["step"]["layers"],
              "passes": {k: v["passes"] for k, v in dx.items()}, "min": {k: v["min"] for k, v in dx.items()},
              "failing_count": {k: v["failing_count"] for k, v in dx.items()}}
    except (KeyError, TypeError) as e:
        return {"9_dx": {"ok": None, "not_read": "harness error"}}, f"harness error: the dX record: {type(e).__name__}: {e}"
    if not a8.get("heldout"):
        return {"9_dx": r9, "10_heldout": {"ok": None, "not_read": "no held-out read"}}, \
            "harness error: an Amendment 8 box without its registered held-out read (--heldout)"
    try:
        h = box["heldout"]
        r10 = {"ok": None, "band": HELDOUT_BAND, "one_sided": True, "loss_before": h["loss_before"],
               "loss_after_step0": h["loss_after_step0"], "loss_after": h["loss_after"], "rel_change_step0": h["rel_change_step0"],
               "rel_change": h["rel_change"], "all_finite": h["all_finite"], "tokens_sha256": h["tokens_sha256"],
               "reported_train_rows": h.get("train")}
    except (KeyError, TypeError) as e:
        return {"9_dx": r9, "10_heldout": {"ok": None, "not_read": "harness error"}}, \
            f"harness error: the held-out record: {type(e).__name__}: {e}"
    if HELDOUT_BAND is None:
        return {"9_dx": r9, "10_heldout": {**r10, "not_read": "Amendment 8's band is not registered"}}, \
            "not registered: Amendment 8's held-out band"
    cap = h["loss_before"] * (1 + HELDOUT_BAND)
    r10["ok"] = h["all_finite"] is True and h["loss_after_step0"] <= cap and h["loss_after"] <= cap
    return {"9_dx": r9, "10_heldout": r10}, None


def accumulation(box):
    """Rule 8 (Amendment 6) and its instrument check: ``(rule record, the VOID's reason or None)``."""
    if GA_BAR is None:
        return {"ok": None, "not_read": "Amendment 6's bar is not registered"}, "not registered: Amendment 6's accumulation bar"
    try:
        r8 = {k: box[k]["rule8"] for k in ("step", "step1")}
        steps = {k: v["check"] for k, v in r8.items()}
        floor = {k: v["floor"]["max_rel"] for k, v in r8.items()}
        modes = {k: (v["algorithms"], v["restored"], box[k]["algorithms"]) for k, v in r8.items()}
        ab = {k: box[k]["ga_check"] for k in ("step", "step1")}
        step_own = {k: box[k]["ga_mb_check"] for k in ("step", "step1")}
        plants = {k: box["ga_plants"][k]["vs_micro_batches"] for k in GA_PLANTS}
        over = {k: v["max_rel"] for k, v in steps.items() if not (v["max_rel"] is not None and v["max_rel"] <= GA_BAR)}
        zero = {k: v["zero_mismatch"] for k, v in steps.items() if v["zero_mismatch"]}
        rec = {"ok": not over and not zero, "bar": GA_BAR, "over": over, "zero_mismatch": zero,
               **{f: {k: v[f] for k, v in steps.items()} for f in ("max_rel", "max_rel_tensor", "cosine", "norm_ratio", "compared")},
               "shape_b": box["shape_b"], "plants": {k: {f: plants[k][f] for f in ("max_rel", "cosine", "norm_ratio")} for k in GA_PLANTS},
               "floor": floor, "modes": modes,
               "reported_step_vs_reference": {k: {f: v[f] for f in ("max_rel", "cosine")} for k, v in step_own.items()},
               "reported_a_vs_b": {k: {f: v[f] for f in ("max_rel", "rel_p50", "cosine", "norm_ratio")} for k, v in ab.items()},
               "reported_route_flips": {f: box["route_flips"][f] for f in ("flipped", "token_layers", "flipped_fraction")}}
    except (KeyError, TypeError) as e:
        return {"ok": None, "not_read": "harness error"}, f"harness error: shape B's record: {type(e).__name__}: {e}"
    for k, (ran, restored, step_mode) in modes.items():
        if (ran, restored, step_mode) != ("deterministic", "default", "default"):
            return rec, (f"instrument: rule 8's check at {k} ran {ran!r}, restored {restored!r}, and the step ran {step_mode!r} "
                         "(expected deterministic / default / default)")
    for k, f in floor.items():
        if f is None or f > GA_BAR / 3:
            return rec, f"instrument: rule 8's floor at {k} is {f!r}, above BAR / 3 = {GA_BAR / 3:.4g}: the check cannot be read"
    for kind in GA_PLANTS:
        if not plants[kind]["max_rel"] > GA_BAR:
            return rec, (f"instrument: the {kind} accumulation plant passes the bar (max_rel {plants[kind]['max_rel']:.3g} <= "
                         f"{GA_BAR}): the bar cannot see it")
    return rec, None


def interior(n):
    """Amendment 3's interior class: the 30B calibration's layers 4-44 of 48, scaled to n layers (8-86 of 94)."""
    return range(-(-4 * n // 48), 44 * n // 48 + 1)


def _sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with open(path) as f:
        return f.read()


def void_cause(run_dir, run, code):
    """Why the run gives no reading, or None when it does (exit 0, or 12: a STRICT failure, which is a reading)."""
    if run.get("mode") != "science" or run.get("science_eligible") is not True:
        return f"not a science run (mode {run.get('mode')!r})"
    if code in REFUSAL_CODES:
        refusal = _read(os.path.join(run_dir, "REFUSAL")).strip() if os.path.exists(os.path.join(run_dir, "REFUSAL")) else "unnamed"
        return f"refusal: {refusal} (exit {code})"
    if os.path.exists(os.path.join(run_dir, "host-fault.json")):
        return "host fault: " + _read(os.path.join(run_dir, "host-fault.json")).strip()[:200]
    summary = _read(os.path.join(run_dir, "summary.txt")) if os.path.exists(os.path.join(run_dir, "summary.txt")) else ""
    if code not in ("0", "12"):
        if "deadline reserve" in summary:
            return f"time-left guard (exit {code})"
        return f"harness error (exit {code})"
    return None


def instrument_void(rd, rep, ref_dir, mask, compare, n):
    """Amendment 3's instrument checks, from the repeat's records; returns (cause or None, the record)."""
    L = n // 2
    null = compare(os.path.join(rd, "repeat"), os.path.join(rd, "box"), mask)        # the null against the box
    rec = {"null_vs_box_max": max(null["rel_err"]), "null_floor": NULL_FLOOR,
           "null_bitwise": rep.get("layer_sha256") == json.loads(_read(os.path.join(rd, "box", "box.json"))).get("layer_sha256"),
           "plants": {}}
    cause = None if rec["null_vs_box_max"] <= NULL_FLOOR else f"the null is not the box (max {rec['null_vs_box_max']:.3g} > {NULL_FLOOR})"
    for kind in PLANTS:
        d = os.path.join(rd, "repeat", f"plant_{kind}")
        pr = json.loads(_read(os.path.join(d, "box.json")))
        u = compare(d, ref_dir, mask)["update_rel_err"]
        pre = compare(d, os.path.join(rd, "repeat"), mask)["rel_err"][:L]      # each earlier layer against the null
        sha, sha_null = pr["layer_sha256"], rep["layer_sha256"]
        p = {"layer": pr["plant"]["layer"], "u_at_layer": u[L], "u_after": u[L + 1:], "fails_band": u[L] > BAND,
             "pre_plant_max_rel_diff": max(pre, default=0.0), "pre_plant_within_floor": all(x <= NULL_FLOOR for x in pre),
             "pre_plant_bitwise": sha[:L] == sha_null[:L], "planted_layer_differs": sha[L] != sha_null[L]}
        rec["plants"][kind] = p
        why = (f"the {kind} plant is not at layer {L}" if pr["plant"] != {"kind": kind, "layer": L} else
               f"the {kind} plant touched the layers before it (max {p['pre_plant_max_rel_diff']:.3g} > {NULL_FLOOR})"
               if not p["pre_plant_within_floor"] else
               f"the {kind} plant did not engage" if not p["planted_layer_differs"] else
               f"the {kind} plant passes the band (u {u[L]:.3g} <= {BAND}): the band cannot see it" if not p["fails_band"] else None)
        cause = cause or why
    return cause, rec


def judge(run_dir, ref_dir):
    gate1, boxmod, compare = _sibling("q235_gate1"), _sibling("q235_box"), _sibling("q235_compare").compare
    out = {"run_dir": os.path.basename(os.path.abspath(run_dir)), "rules": {}, "reported": {}}
    rd = os.path.join(run_dir, "receipts")
    try:
        run = json.loads(_read(os.path.join(rd, "run.json")))
        codes = [f for f in os.listdir(run_dir) if f.startswith("TC1_EXIT_CODE.")]
        if len(codes) != 1:
            raise ValueError(f"{len(codes)} exit-code records")
        code = _read(os.path.join(run_dir, codes[0])).strip()
        out["exit_code"] = code
        cause = void_cause(run_dir, run, code)
        if cause:
            out.update(verdict="VOID", why=cause)
            return out
        card = _read(os.path.join(rd, "card.txt")).split(",")[0].strip()
        box = json.loads(_read(os.path.join(rd, "box", "box.json")))
        mask = torch.load(os.path.join(rd, "tokens.pt"), weights_only=True)["attention_mask"]
        cmp = compare(os.path.join(rd, "box"), ref_dir, mask)
        n = cmp["layers"]
        stopped = code == "12"                                  # the STRICT failure stops the run before the repeat
        if not stopped:
            rep = json.loads(_read(os.path.join(rd, "repeat", "box.json")))
            instrument, out["instrument"] = instrument_void(rd, rep, ref_dir, mask, compare, n)
    except (OSError, KeyError, ValueError, AssertionError, RuntimeError, IndexError, TypeError) as e:
        out.update(verdict="VOID", why=f"harness error: {type(e).__name__}: {e}")
        return out
    losses = [box.get(k, {}).get("loss") for k in ("step", "step1")]
    strict = boxmod.strict_failures(box)
    u = cmp["update_rel_err"]
    over = [i for i in interior(n) if not u[i] <= BAND]
    r = out["rules"]
    # exit 12 is the box's STRICT failure after both steps ran: the step completed, and rules 3-5 carry the failure
    r["1_completed_on_class"] = {"ok": code in ("0", "12") and bool(re.match(CARD_RE, card)), "card": card, "exit_code": code}
    r["2_loss_finite"] = {"ok": all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses), "losses": losses}
    r["3_5_strict"] = {"ok": not strict, "failures": strict}
    if stopped:
        r["6_repeat"] = r["7_band"] = {"ok": None, "not_read": "exit 12: the run stopped at the STRICT failure, before the repeat"}
    else:
        repeat_ok, repeat_rec = gate1.repeat_within_tolerance(box, rep["repeat"]["loss"])
        r["6_repeat"] = {"ok": repeat_ok, **repeat_rec}
        r["7_band"] = {"ok": not over, "band": BAND, "interior": [interior(n)[0], interior(n)[-1]], "layers_over": over,
                       "u_interior_max": max(u[i] for i in interior(n))}
    ga_cause = l2_cause = None
    if "shape_b" in box:                                       # Amendment 6: both steps ran, so rule 8 is read even after exit 12
        r["8_accumulation"], ga_cause = accumulation(box)
    if box.get("amendment8"):                                  # Amendment 8: both steps ran, so rules 9 and 10 are read after exit 12
        l2, l2_cause = l2_fields(box)
        r.update(l2)
    steps = [box.get(k, {}) for k in ("step", "step1")]
    peak = max((s.get("max_memory_allocated") or 0) for s in steps) / (1 << 30)
    out["reported"] = {
        "u_profile": u, "full_state": {"rel_err": cmp["rel_err"], "median_upper_middle": cmp["rel_err_median"], "max": cmp["rel_err_max"]},
        "router_overlap_min": cmp["router_overlap_min"], "router_set_agree_min": cmp["router_set_agree_min"], "loss_gap": cmp["loss_gap"],
        "device_peak_gib": round(peak, 2), "estimate_device_gib": ESTIMATE_DEVICE_GIB,
        "device_deviation": round(peak / ESTIMATE_DEVICE_GIB - 1, 3),
        "estimator_issue": abs(peak / ESTIMATE_DEVICE_GIB - 1) > 0.15,
        "h2d_gib_per_step": [round((s.get("h2d_bytes_offload") or 0) / (1 << 30), 2) for s in steps],
        "estimate_link_gib": ESTIMATE_LINK_GIB, "step_elapsed_s": [s.get("elapsed_s") for s in steps]}
    failed = [k for k, v in r.items() if v["ok"] is False]
    cause = (f"instrument: {instrument}" if not stopped and instrument else None) or ga_cause or l2_cause
    if cause:
        out.update(verdict="VOID", why=cause + (f" (rules also failing: {', '.join(failed)})" if failed else ""))
        return out
    held = "1-" + ("8" if "8_accumulation" in r else "7") + (", 9-10" if "10_heldout" in r else "")
    held = "1-10" if held == "1-8, 9-10" else held
    out.update(verdict="FAIL" if failed else "PASS", why=("rules failing: " + ", ".join(failed)) if failed else f"rules {held} hold")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = judge(a.run_dir, a.ref)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(json.dumps({"verdict": r["verdict"], "why": r["why"]}))
    sys.exit(EXIT[r["verdict"]])


if __name__ == "__main__":
    main()
