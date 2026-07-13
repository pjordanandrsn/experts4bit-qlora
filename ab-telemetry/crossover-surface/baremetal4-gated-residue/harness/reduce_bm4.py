#!/usr/bin/env python3
"""BM4 reducer: arm table + frozen K1-K4 / R3-R5 / G2 verdicts (prereg_baremetal4.json).

Verdict rules are transcribed from the prereg, committed+OTS-stamped before
provisioning; this script only mechanizes them. stdout = bm4_results.json.
"""
import json
import re
import statistics
import sys
from pathlib import Path

DQ = Path(sys.argv[1])
PROBE = Path(sys.argv[2])

ARMS = [
    "warmup", "r64_noprobe", "r64_probe", "v64",
    "off_f1p0_a", "off_f1p0_b", "off_f0p5", "off_f0p0", "cent_off",
    "on_f1p0_a", "on_f1p0_b", "on_f0p9375", "on_f0p875", "on_f0p75",
    "on_f0p5", "on_f0p0", "cent_on",
    "v512", "v2048", "r512_probe", "r2048_probe_a", "r2048_probe_b",
]


def sps(name):
    log = DQ / f"out_{name}.log"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    m = re.findall(r"'train_steps_per_second': '?([0-9.]+)'?", text)
    if not m:
        return None
    nloss = len(re.findall(r"'loss':", text))
    if nloss < 30:
        return None  # crashed arm mimicking a result (the tqdm trap)
    return float(m[-1])


def stepsec(name):
    v = sps(name)
    return None if v in (None, 0) else 1.0 / v


def stats_line(name):
    log = DQ / f"out_{name}.log"
    if not log.exists():
        return None
    m = re.findall(r"E4B_PREFETCH_STATS (\{.*\})", log.read_text(errors="replace"))
    return json.loads(m[-1]) if m else None


t = {a: stepsec(a) for a in ARMS}
out = {"arms_s_per_step": {k: (round(v, 4) if v else None) for k, v in t.items()}}
out["prefetch_stats"] = {a: stats_line(a) for a in ARMS if a.startswith(("on_", "cent_on"))}


def mean(*vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


ON1 = mean(t["on_f1p0_a"], t["on_f1p0_b"])
OFF1 = mean(t["off_f1p0_a"], t["off_f1p0_b"])
verdicts = {}
if ON1 and OFF1:
    out["floors"] = {
        "on_f1_spread": round(abs(t["on_f1p0_a"] - t["on_f1p0_b"]), 4),
        "off_f1_spread": round(abs(t["off_f1p0_a"] - t["off_f1p0_b"]), 4),
    }
    # K1: ON at f=0.9375 within 5% of ON floor
    if t["on_f0p9375"]:
        k1 = t["on_f0p9375"] / ON1 - 1
        verdicts["K1"] = {"on_f0p9375_vs_floor_pct": round(k1 * 100, 1), "pass": k1 <= 0.05}
    # K2: ON<=OFF at every interior f; at f=0.5 removes >=25% of OFF penalty
    interior = {}
    ok_all = True
    for f in ("0p9375", "0p875", "0p75", "0p5"):
        on, off = t.get(f"on_f{f}"), t.get(f"off_f{f}")
        if on and off:
            interior[f] = round((on / off - 1) * 100, 1)
            ok_all &= on <= off * 1.0
    removal = None
    if t.get("on_f0p5") and t.get("off_f0p5"):
        pen_off = t["off_f0p5"] - OFF1
        pen_on = t["on_f0p5"] - ON1
        removal = (pen_off - pen_on) / pen_off if pen_off > 0 else None
    verdicts["K2"] = {
        "on_vs_off_interior_pct": interior,
        "f0p5_penalty_removal_pct": round(removal * 100, 1) if removal is not None else None,
        "pass": bool(ok_all and removal is not None and removal >= 0.25),
    }
    # K3: centroid ON <= OFF; f=1.0 inertness within 3%
    inert = ON1 / OFF1 - 1
    k3c = (t.get("cent_on") or 9e9) <= (t.get("cent_off") or 0)
    verdicts["K3"] = {
        "cent_on": t.get("cent_on"), "cent_off": t.get("cent_off"),
        "f1_inert_pct": round(inert * 100, 2),
        "pass": bool(k3c and abs(inert) <= 0.03),
    }
    # K4: all-flash ON <= OFF (run rule preserves the v3 win)
    if t.get("on_f0p0") and t.get("off_f0p0"):
        k4 = t["on_f0p0"] / t["off_f0p0"] - 1
        verdicts["K4"] = {"on_vs_off_f0_pct": round(k4 * 100, 1), "pass": k4 <= 0.0}

# G2 perf-degenerate: fused-OFF f=1.0 within 0.5% of public RAMStore (r64_noprobe)
if OFF1 and t.get("r64_noprobe"):
    g2 = OFF1 / t["r64_noprobe"] - 1
    verdicts["G2"] = {"fusedoff_f1_vs_ramstore_pct": round(g2 * 100, 2), "pass": abs(g2) <= 0.005}

# R3: residue reproduction at seq2048 (factor 2 of 102 ms)
R2048 = mean(t.get("r2048_probe_a"), t.get("r2048_probe_b"))
if R2048 and t.get("v2048"):
    res = {
        64: (mean(t.get("r64_noprobe"), t.get("r64_probe")) or 0) - (t.get("v64") or 0),
        512: (t.get("r512_probe") or 0) - (t.get("v512") or 0),
        2048: R2048 - t["v2048"],
    }
    out["residue_ms"] = {k: round(v * 1e3, 1) for k, v in res.items()}
    verdicts["R3"] = {
        "residue_2048_ms": round(res[2048] * 1e3, 1),
        "pass": 51 <= res[2048] * 1e3 <= 204,
    }

# R4: probe overhead (r64 probe vs noprobe within 3%)
if t.get("r64_probe") and t.get("r64_noprobe"):
    r4 = t["r64_probe"] / t["r64_noprobe"] - 1
    verdicts["R4"] = {"probe_overhead_pct": round(r4 * 100, 2), "pass": abs(r4) <= 0.03}


# R5: probe medians (steps 10..40 approximated by dropping the first quarter of records)
def probe_medians(path):
    if not path.exists():
        return None
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if "submit_ms" in r]
    if len(recs) < 100:
        return None
    recs = recs[len(recs) // 4:]
    sub = statistics.median(r["submit_ms"] for r in recs)
    dev = statistics.median(r["dev_ms"] / (r["mb"] / 1e3) for r in recs if r["mb"])
    mems = [r["mem"] for r in recs if "mem" in r]
    return {
        "n": len(recs),
        "submit_ms_med": round(sub, 3),
        "dev_ms_per_gb_med": round(dev, 2),
        "alloc_retries_span": (
            [mems[0].get("num_alloc_retries"), mems[-1].get("num_alloc_retries")] if mems else None
        ),
    }


pm = {
    "r64": probe_medians(PROBE / "r64.jsonl"),
    "r512": probe_medians(PROBE / "r512.jsonl"),
    "r2048_a": probe_medians(PROBE / "r2048_a.jsonl"),
    "r2048_b": probe_medians(PROBE / "r2048_b.jsonl"),
}
out["probe_medians"] = pm
if pm["r64"] and pm["r2048_a"]:
    r2 = pm["r2048_b"] or pm["r2048_a"]
    sub_ratio = mean(pm["r2048_a"]["submit_ms_med"], r2["submit_ms_med"]) / pm["r64"]["submit_ms_med"]
    dev_ratio = mean(pm["r2048_a"]["dev_ms_per_gb_med"], r2["dev_ms_per_gb_med"]) / pm["r64"]["dev_ms_per_gb_med"]
    branch = "H-D"
    if sub_ratio >= 1.5 and dev_ratio < 1.3:
        branch = "H-C"
    elif dev_ratio >= 1.5 and sub_ratio < 1.3:
        branch = "H-B"
    elif sub_ratio >= 1.5 and dev_ratio >= 1.5:
        branch = "BOTH"
    verdicts["R5"] = {
        "submit_ratio_2048v64": round(sub_ratio, 2),
        "dev_per_gb_ratio_2048v64": round(dev_ratio, 2),
        "branch": branch,
    }

out["verdicts"] = verdicts
kline = " ".join(
    f"{k}={'PASS' if v.get('pass') else v.get('branch', 'FAIL')}" for k, v in sorted(verdicts.items())
)
out["verdict_line"] = kline
print(json.dumps(out, indent=1))
print(f"REDUCE OK: {kline}", file=sys.stderr)
