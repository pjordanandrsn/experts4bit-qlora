# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's milestone verdict, applied mechanically from the registered decision rules (bench/q235/PREREG-q235.md, "Decision rules for
the eventual box"), never read by eye.

Inputs: the box's fetched run directory (``q235_run.sh``'s layout: ``receipts/run.json``, ``receipts/card.txt``, ``receipts/box/``,
``receipts/repeat/``, ``receipts/tokens.pt``, ``TC1_EXIT_CODE.<nonce>``, ``REFUSAL``, ``summary.txt``) and the full-depth
layer-streamed bf16 reference made off the box (``layer_stream_ref.py``, the same tokens).

**VOID** (no reading) when the run is not a science run, or it ended in a refusal (a named host floor: exit 13, 14, 17 or 18 with its
``REFUSAL``), a host fault (``host-fault.json``), a time-left guard (the deadline reserve) or a harness error (any other exit but the
STRICT failure's 12, or a record missing).

**MILESTONE PASS** iff all of these hold, else **FAIL** naming each rule that fails:
1. the step completed (exit 0, or 12 when a STRICT rule failed after both steps) on the registered class (``receipts/card.txt``);
2. every step's loss is finite;
3-5. Amendment 2's STRICT rules and the frozen base, re-derived from ``box.json`` (``q235_box.strict_failures``, fail-closed);
6. the fresh-process repeat of step 0 within the larger of the in-process spread and 1e-4 relative (a miss is a finding with its size);
7. Amendment 1's bands against the reference (``q235_gate1.band_failures``).

**Reported, not gating:** the device peak against the registered estimate (a deviation above 15 % names an estimator issue), the bytes
moved per step against the estimate's link line, and the steps' elapsed time.

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
ESTIMATE_DEVICE_GIB = 24.76                    # e4b's estimate for the registered shape (bf16 adapters, 8 x 331 tokens; 24140cd7)
ESTIMATE_LINK_GIB = 237.94                     # the estimate's link line per micro-batch (PREREG "Refusal points" 6)
EXIT = {"PASS": 0, "FAIL": 1, "VOID": 2}
REFUSAL_CODES = {"13", "14", "17", "18"}


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


def judge(run_dir, ref_dir):
    gate1, boxmod, compare = _sibling("q235_gate1"), _sibling("q235_box"), _sibling("q235_compare").compare
    out = {"run_dir": os.path.basename(os.path.abspath(run_dir)), "rules": {}, "reported": {}}
    try:
        run = json.loads(_read(os.path.join(run_dir, "receipts", "run.json")))
        codes = [f for f in os.listdir(run_dir) if f.startswith("TC1_EXIT_CODE.")]
        if len(codes) != 1:
            raise ValueError(f"{len(codes)} exit-code records")
        code = _read(os.path.join(run_dir, codes[0])).strip()
        out["exit_code"] = code
        cause = void_cause(run_dir, run, code)
        if cause:
            out.update(verdict="VOID", why=cause)
            return out
        card = _read(os.path.join(run_dir, "receipts", "card.txt")).split(",")[0].strip()
        box = json.loads(_read(os.path.join(run_dir, "receipts", "box", "box.json")))
        rep = json.loads(_read(os.path.join(run_dir, "receipts", "repeat", "box.json")))
        mask = torch.load(os.path.join(run_dir, "receipts", "tokens.pt"), weights_only=True)["attention_mask"]
        cmp = compare(os.path.join(run_dir, "receipts", "box"), ref_dir, mask)
    except (OSError, KeyError, ValueError, AssertionError, RuntimeError) as e:
        out.update(verdict="VOID", why=f"harness error: {type(e).__name__}: {e}")
        return out
    losses = [box.get(k, {}).get("loss") for k in ("step", "step1")]
    strict = boxmod.strict_failures(box)
    repeat_ok, repeat_rec = gate1.repeat_within_tolerance(box, rep["repeat"]["loss"])
    bands = gate1.band_failures(cmp)
    r = out["rules"]
    # exit 12 is the box's STRICT failure after both steps ran: the step completed, and rules 3-5 carry the failure
    r["1_completed_on_class"] = {"ok": code in ("0", "12") and bool(re.match(CARD_RE, card)), "card": card, "exit_code": code}
    r["2_loss_finite"] = {"ok": all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses), "losses": losses}
    r["3_5_strict"] = {"ok": not strict, "failures": strict}
    r["6_repeat"] = {"ok": repeat_ok, **repeat_rec}
    r["7_bands"] = {"ok": not bands, "failures": bands, "rel_err_median": cmp["rel_err_median"], "rel_err_max": cmp["rel_err_max"],
                    "router_overlap_min": cmp["router_overlap_min"], "router_set_agree_min": cmp["router_set_agree_min"],
                    "loss_gap": cmp["loss_gap"]}
    steps = [box.get(k, {}) for k in ("step", "step1")]
    peak = max((s.get("max_memory_allocated") or 0) for s in steps) / (1 << 30)
    out["reported"] = {
        "device_peak_gib": round(peak, 2), "estimate_device_gib": ESTIMATE_DEVICE_GIB,
        "device_deviation": round(peak / ESTIMATE_DEVICE_GIB - 1, 3),
        "estimator_issue": abs(peak / ESTIMATE_DEVICE_GIB - 1) > 0.15,
        "h2d_gib_per_step": [round((s.get("h2d_bytes_offload") or 0) / (1 << 30), 2) for s in steps],
        "estimate_link_gib": ESTIMATE_LINK_GIB, "step_elapsed_s": [s.get("elapsed_s") for s in steps]}
    failed = [k for k, v in r.items() if not v["ok"]]
    out.update(verdict="FAIL" if failed else "PASS",
               why=("rules failing: " + ", ".join(failed)) if failed else "rules 1-7 hold")
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
