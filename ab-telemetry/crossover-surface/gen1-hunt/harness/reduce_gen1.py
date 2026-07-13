#!/usr/bin/env python3
"""gen1 reducer: P1-P5 verdicts per prereg_gen1_hunt.json (frozen before provisioning)."""
import json
import re
import sys
from pathlib import Path

DQ = Path(sys.argv[1])
L = float(sys.argv[2])

ARMS = ["warmup", "v_a", "r_a", "v_b", "r_b", "fv000", "fv025", "fv050", "fv075", "fv100"]

# frozen anchors (H100 chain, prereg acabdfc lineage)
GB_PER_STEP = 6.44
V_H100 = 0.2075
HIDDEN_H100 = 0.092
VALUE_3090 = 0.0395  # s/GB at L=20.36


def stepsec(name):
    log = DQ / f"out_{name}.log"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    m = re.findall(r"'train_steps_per_second': '?([0-9.]+)'?", text)
    if not m:
        return None
    if len(re.findall(r"'loss':", text)) < 30:
        return None
    return 1.0 / float(m[-1])


t = {a: stepsec(a) for a in ARMS}
out = {"L_gbps": L, "arms_s_per_step": {k: (round(v, 4) if v else None) for k, v in t.items()}}


def mean(*v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


V = mean(t["v_a"], t["v_b"])
R = mean(t["r_a"], t["r_b"])
verdicts = {}
if V and R:
    out["floors"] = {
        "v_spread": round(abs(t["v_a"] - t["v_b"]), 4),
        "r_spread": round(abs(t["r_a"] - t["r_b"]), 4),
    }
    tax = R - V
    out["tax_s"] = round(tax, 4)
    # P1 ends parity
    p1 = {}
    if t["fv100"]:
        p1["fv1_vs_V_pct"] = round((t["fv100"] / V - 1) * 100, 1)
    if t["fv000"]:
        p1["fv0_vs_R_pct"] = round((t["fv000"] / R - 1) * 100, 1)
    verdicts["P1"] = {
        **p1,
        "pass": all(abs(x) <= 3.0 for x in p1.values()) if len(p1) == 2 else False,
    }
    # P2 interior linearity vs t(fv) = V + (1-fv)(R-V)
    p2 = {}
    for fv, arm in ((0.25, "fv025"), (0.5, "fv050"), (0.75, "fv075")):
        if t[arm]:
            pred = V + (1 - fv) * (R - V)
            p2[arm] = round((t[arm] / pred - 1) * 100, 1)
    verdicts["P2"] = {
        "err_pct": p2, "pass": bool(p2) and all(abs(x) <= 10.0 for x in p2.values()),
    }
    # P3 frozen tax model
    tax_pred = max(0.0, GB_PER_STEP / L - HIDDEN_H100 * (V / V_H100))
    ratio = tax / tax_pred if tax_pred > 0 else None
    verdicts["P3"] = {
        "tax_pred_s": round(tax_pred, 4), "tax_meas_s": round(tax, 4),
        "ratio": round(ratio, 2) if ratio else None,
        "pass": ratio is not None and 0.5 <= ratio <= 2.0,
    }
    # P4 half residency recovers 40-60%
    if t["fv050"]:
        rec = (R - t["fv050"]) / tax if tax > 0 else None
        verdicts["P4"] = {
            "fv05_recovery_pct": round(rec * 100, 1) if rec is not None else None,
            "pass": rec is not None and 0.40 <= rec <= 0.60,
        }
    # P5 dose response: per-GB promotion value vs the 3090's
    value = tax / GB_PER_STEP
    out["per_gb_value_s"] = round(value, 4)
    verdicts["P5"] = {
        "per_gb_value_s": round(value, 4),
        "vs_3090_x": round(value / VALUE_3090, 2),
        "pass": (value > VALUE_3090) if L < 15 else True,
    }

out["verdicts"] = verdicts
line = " ".join(f"{k}={'PASS' if v.get('pass') else 'FAIL'}" for k, v in sorted(verdicts.items()))
out["verdict_line"] = f"L={L} {line}"
print(json.dumps(out, indent=1))
print(f"REDUCE OK: {out['verdict_line']}", file=sys.stderr)
