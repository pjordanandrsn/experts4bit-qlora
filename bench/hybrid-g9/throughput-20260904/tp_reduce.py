#!/usr/bin/env python3
"""Reduce the throughput-parity lanes' box_out dirs into one table.
Usage: tp_reduce.py <box_out dir>... ; the Qwen3 rows are the reference."""
import json, os, sys, glob, re
FAMS = [("qwen3", "Qwen3-30B-A3B (reference)"), ("olmoe", "OLMoE-1B-7B"), ("granite", "Granite-3.1-3B-A800M"),
        ("gptoss", "gpt-oss-20b"), ("gemma4", "Gemma-4-26B-A4B"), ("mixtral", "Mixtral-8x7B")]
ARMS = ["b1_nf4", "b1_int4exp", "b1_calib", "b1_fused", "b1_nf4fused", "b16_nf4", "b16_int4exp"]
PPL = ["ppl_nf4", "ppl_int4exp", "ppl_calib", "ppl_fused", "ppl_nf4fused"]
def load(dirs, tag, arm):
    for d in dirs:
        p = os.path.join(d, f"{tag}_{arm}.json")
        if os.path.exists(p):
            try: return json.load(open(p))
            except Exception: return None
    return None
def why_failed(dirs, tag, arm):
    for d in dirs:
        p = os.path.join(d, f"run_{tag}_{arm}.log")
        if os.path.exists(p):
            txt = open(p, errors="replace").read()
            m = re.findall(r"(RuntimeError: [^\n]{0,110}|REFUSED[^\n]{0,110}|OutOfMemoryError[^\n]{0,60}|AssertionError: [^\n]{0,90})", txt)
            if m: return m[-1]
            return "no result (see log)"
    return "not run"
def tok_s(r):
    if not r: return None
    if "aggregate_tok_s" in r: return float(r["aggregate_tok_s"])
    if r.get("step_ms_clean"): return 1000.0 / float(r["step_ms_clean"])
    return None
dirs = sys.argv[1:]
out = ["| family | K8 nll nf4 / int4exp / calib (2048) | B=1 nf4 | B=1 int4exp | B=1 calib | B=1 fused (int4+calib+folds) | B=16 nf4 | B=16 int4exp |", "|---|---|---|---|---|---|---|---|"]
notes = []
ref = {}
for tag, name in FAMS:
    if not any(glob.glob(os.path.join(d, f"{tag}_*.json")) for d in dirs) and not any(glob.glob(os.path.join(d, f"run_{tag}_*.log")) for d in dirs):
        continue
    ppl = []
    for a in ("ppl_nf4", "ppl_int4exp", "ppl_calib"):
        r = load(dirs, tag, a); ppl.append(f"{r['mean_nll']:.4f}" if r and "mean_nll" in r else "—")
    cells = []
    for a in ARMS[:4] + ARMS[5:]:
        r = load(dirs, tag, a); v = tok_s(r)
        if v is None:
            cells.append("✗"); notes.append(f"{tag}/{a}: {why_failed(dirs, tag, a)}")
        else:
            cells.append(f"{v:.0f}")
            if tag == "qwen3": ref[a] = v
    out.append(f"| {name} | {' / '.join(ppl)} | " + " | ".join(cells) + " |")
print("\n".join(out))
if ref:
    print("\nRatio to the Qwen3-30B reference on the same protocol (B=1 nf4 / B=1 best / B=16 nf4):")
    for tag, name in FAMS:
        if tag == "qwen3": continue
        r_nf4, r_b16 = tok_s(load(dirs, tag, "b1_nf4")), tok_s(load(dirs, tag, "b16_nf4"))
        best = max([v for v in (tok_s(load(dirs, tag, a)) for a in ("b1_nf4", "b1_int4exp", "b1_calib", "b1_fused")) if v] or [0])
        qbest = max([v for v in (ref.get(a) for a in ("b1_nf4", "b1_int4exp", "b1_calib", "b1_fused")) if v] or [1])
        if r_nf4: print(f"  {name}: {r_nf4/ref.get('b1_nf4',1):.2f}x / {best/qbest:.2f}x / {(r_b16/ref.get('b16_nf4',1)) if r_b16 and ref.get('b16_nf4') else float('nan'):.2f}x")
if notes:
    print("\nRefusals / failures (the build-out list):")
    for n in notes: print("  - " + n)
