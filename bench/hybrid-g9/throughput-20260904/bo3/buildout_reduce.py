"""Reduce lane logs (run_<fam>_ppl_<arm>.log / run_<fam>_b1_<arm>.log / run_<fam>_b16_<arm>.log) into one
per-family build-out table: K8 nll (nats), delta vs NF4, B=1 ms and tok/s, B=16 agg tok/s, ratios vs NF4.
Usage: buildout_reduce.py <dir> [<dir>...]  (later dirs override earlier ones for the same (fam, arm))."""
import glob
import math
import os
import re
import sys

FLOOR = {"granite": 0.0033, "gptoss": 0.0176, "qwen3": 0.0095, "mixtral": None, "gemma4": None}
NAME = {"granite": "Granite-3.1-3B-A800M", "gptoss": "gpt-oss-20b", "qwen3": "Qwen3-30B-A3B",
        "mixtral": "Mixtral-8x7B-Instruct", "gemma4": "Gemma-4-26B-A4B-it"}
ARMS = ["nf4", "int4exp", "calib", "r1", "r12", "epi", "stack", "all", "all_rope", "head", "allhead",
        "best", "bestdense", "bestdensehead", "calibbias", "r1epicalib", "r1epicalibhead",
        "stack_old", "stackr2_old", "stack_new", "stackr2_new", "mx_int4exp", "mx_stack"]
DESC = {"nf4_r12epi_new": "NF4 experts + r1 + r2 + epilogue, with the rotary-only fold (#379) — Granite's licensed stack",
        "nf4_r12epi_old": "NF4 experts + r1 + r2 + epilogue, before the rotary-only fold",
        "nf4_r1epi_new": "NF4 experts + r1 + epilogue (round-2 off, control)", "nf4_r1epi_old": "NF4 experts + r1 + epilogue (round-2 off, control)","nf4": "NF4 experts, bf16 attention (baseline)", "int4exp": "+ int4 experts",
        "calib": "int4 experts + C4-calibrated int4 attention", "r1": "round-1 norm folds only",
        "r12": "round-1 + round-2 folds only", "epi": "router epilogue only",
        "stack": "int4 experts + r1 + epilogue", "all": "int4 + calib attn + r1 + r2 + epilogue",
        "all_rope": "all + round-2 fold on unfused attention (#375)", "head": "NF4 + calibrated int4 output head only (#373)",
        "allhead": "all + calibrated int4 head", "best": "int4 + calib attn + r1 + epilogue",
        "bestdense": "best + calibrated int4 dense MLP (#378)", "bestdensehead": "bestdense + int4 head",
        "calibbias": "int4 + calib attn with biases (#377)", "r1epicalib": "int4 + calib(bias) + r1 + epi",
        "r1epicalibhead": "r1epicalib + int4 head", "stack_old": "stack, old cut (control)",
        "stackr2_old": "stack + r2, old cut", "stack_new": "stack, new cut (control)",
        "stackr2_new": "stack + r2 + rope-only fold (#379)", "mx_int4exp": "native MXFP4 store (#372)",
        "mx_stack": "MXFP4 store + r1 + epi"}
rows = {}
pat = re.compile(r"run_(?P<fam>[a-z0-9]+)(?P<mx>_mx)?_(?P<kind>ppl|b1|b16)_(?P<arm>[a-z0-9_]+)\.log$")
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "run_*.log"))):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        fam, kind, arm = m.group("fam"), m.group("kind"), m.group("arm")
        txt = open(f, errors="replace").read()
        if m.group("mx") and not arm.startswith("mx_"):
            arm = "mx_" + arm
        if arm.startswith("probe") or arm.startswith("mxh_") or arm.startswith("mx2_"):
            continue                       # probe windows / 16-step runs: not the table's window
        r = rows.setdefault((fam, arm), {})
        if kind == "ppl":
            k = re.findall(r"K8_PPL steps=(\d+) nll=([\d.]+) .*?sha=([0-9a-f]+)", txt)
            if k:
                r["ppl"] = float(k[-1][1]); r["steps"] = int(k[-1][0]); r["sha"] = k[-1][2]
        elif kind == "b1":
            k = re.findall(r"B1D_TIMED_\w+ steps=\d+ step=([\d.]+)ms", txt)
            if k:
                r["b1_ms"] = float(k[-1])
        else:
            k = re.findall(r"BV3_GRAPH batch=16 steps=\d+ step=[\d.]+ms agg=([\d.]+)tok/s", txt)
            if k:
                r["b16"] = float(k[-1])
        if "REFUSED" in txt or "RuntimeError" in txt:
            r.setdefault("refused", []).append(kind)
for fam in ["qwen3", "granite", "gemma4", "mixtral", "gptoss"]:
    fams = {a: r for (f, a), r in rows.items() if f == fam}
    if not fams:
        continue
    base = fams.get("nf4", {})
    floor = FLOOR[fam]
    print(f"\n### {NAME[fam]}" + (f" — arithmetic-order floor {floor} nats" if floor else " — no stable floor (see SERVING-PARITY)"))
    print("| arm | configuration | K8 nll | Δ nats | Δ ppl | gate (k8_gate: uncal |Δppl|≤0.05; calib ≤+0.05 one-sided) | B=1 ms | B=1 tok/s | ×NF4 | B=16 tok/s | ×NF4 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS + sorted(a for a in fams if a not in ARMS):
        r = fams.get(arm)
        if not r:
            continue
        ppl = r.get("ppl"); d = (ppl - base["ppl"]) if (ppl is not None and "ppl" in base) else None
        b1 = r.get("b1_ms"); tps = 1000 / b1 if b1 else None
        x1 = (base["b1_ms"] / b1) if (b1 and "b1_ms" in base) else None
        b16 = r.get("b16"); x16 = (b16 / base["b16"]) if (b16 and "b16" in base) else None
        ref = " REFUSED(" + ",".join(r["refused"]) + ")" if r.get("refused") else ""
        fmt = lambda v, p: ("—" if v is None else f"{v:.{p}f}")
        dppl = (math.exp(ppl) - math.exp(base["ppl"])) if d is not None else None
        calibrated = any(k in arm for k in ("calib", "all", "head", "best"))
        if dppl is None:
            verdict = "—"
        elif fam == "gemma4":
            verdict = "no instrument (chaos band)"
        elif fam == "gptoss":
            verdict = "no instrument (OOD regime, ppl ≈ 560)"
        elif arm == "nf4":
            verdict = "baseline"
        elif calibrated:
            verdict = "pass*" if dppl <= 0.05 else "FAIL"
        else:
            verdict = "pass" if abs(dppl) <= 0.05 else "FAIL"
        print(f"| `{arm}` | {DESC.get(arm, arm)}{ref} | {fmt(ppl,5)} | {fmt(d,4)} | {fmt(dppl,4)} | {verdict} | {fmt(b1,2)} | {fmt(tps,1)} | {fmt(x1,2)} | {fmt(b16,1)} | {fmt(x16,2)} |")
print("\n`pass*` = one-sided calibrated rule on ONE text; the second, out-of-domain text has not been scored on these lanes.")
