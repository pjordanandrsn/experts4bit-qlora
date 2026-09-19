#!/usr/bin/env python3
"""bench/p46/p46_reduce.py -- read P46 (the adapter path) from the four arm receipts.

    python bench/p46/p46_reduce.py <run-dir>/tp4 [--md RESULTS-p46.md] [--json out.json]

Rules (P46-PREREG.md, in code): P1 auto takes the loop every step (lora_path_loop >= 1 and lora_path_padded == 0 in every
step's census); P2 padded s/step <= 0.55x auto (refuted > 0.8x); P3 grouped_mm refuses (NOT_READ) or <= 0.45x; P4 parity vs
reference_attn4: |delta held-out final| <= 0.05 and median per-step |delta loss| <= 0.05; P5 padded peak VRAM <= 1.4x auto.
A missing receipt is NOT_READ, never a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

ARMS = {"reference": "qwen3_e4b_reference_attn4", "auto": "qwen3_e4b_fused_attn4", "padded": "qwen3_e4b_fused_attn4_pad",
        "grouped_mm": "qwen3_e4b_fused_attn4_gmm"}
BAND = 0.05


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def path_census(cell: dict) -> dict:
    """Per-step lora_path_* deltas from kernel_calls_all (each entry is after-before for one step)."""
    steps = cell.get("kernel_calls_all") or []
    keys = sorted({k for s in steps for k in s if k.startswith("lora_path_")})
    per = {k: [int(s.get(k, 0)) for s in steps] for k in keys}
    return {"steps": len(steps), "per_step": per,
            "loop_every_step": bool(steps) and all(s.get("lora_path_loop", 0) >= 1 for s in steps),
            "padded_never": all(s.get("lora_path_padded", 0) == 0 for s in steps),
            "grouped_mm_every_step": bool(steps) and all(s.get("lora_path_grouped_mm", 0) >= 1 for s in steps)}


def parity(ref: dict, arm: dict) -> dict:
    d_final = arm["eval_loss_final"] - ref["eval_loss_final"]
    n = min(len(ref["losses"]), len(arm["losses"]))
    med = statistics.median(abs(a - b) for a, b in zip(arm["losses"][:n], ref["losses"][:n])) if n else None
    ok = abs(d_final) <= BAND and med is not None and med <= BAND
    return {"delta_final": round(d_final, 5), "median_step_abs_delta": None if med is None else round(med, 5), "band": BAND, "passes": ok}


def reduce(d: str) -> dict:
    cells = {k: _load(os.path.join(d, v + ".json")) for k, v in ARMS.items()}
    out = {"arms": {}, "verdicts": {}}
    for k, c in cells.items():
        if c is None:
            out["arms"][k] = {"status": "MISSING"}
            continue
        out["arms"][k] = {"status": c.get("status"), "s_per_step_median_11plus": c.get("s_per_step_median_11plus"), "peak_vram_gb": c.get("peak_vram_gb"),
                          "eval_loss_final": c.get("eval_loss_final"), "tokens_per_s": c.get("tokens_per_s"), "path_census": path_census(c) if c.get("kernel_calls_all") else None,
                          "reason": c.get("reason")}
    ref, auto, pad, gmm = (cells[k] for k in ("reference", "auto", "padded", "grouped_mm"))
    v = out["verdicts"]
    if auto and auto.get("status") == "ok":
        pc = path_census(auto)
        v["P1"] = {"loop_every_step": pc["loop_every_step"], "padded_never": pc["padded_never"],
                   "verdict": "HOLDS" if pc["loop_every_step"] and pc["padded_never"] else "REFUTED"}
    else:
        v["P1"] = {"verdict": "NOT_READ (no ok auto arm)"}
    def speed(cand, name, hold, refute):
        if not (auto and cand and auto.get("status") == "ok" and cand.get("status") == "ok"):
            why = (cand or {}).get("status") or "missing"
            return {"verdict": f"NOT_READ ({name} arm {why})", "reason": (cand or {}).get("reason")}
        r = cand["s_per_step_median_11plus"] / auto["s_per_step_median_11plus"]
        return {"ratio_to_auto": round(r, 4), "s_per_step": cand["s_per_step_median_11plus"], "auto_s_per_step": auto["s_per_step_median_11plus"],
                "verdict": "HOLDS" if r <= hold else "REFUTED" if r > refute else f"INCONCLUSIVE ({hold}..{refute})"}
    v["P2"] = speed(pad, "padded", 0.55, 0.8)
    v["P3"] = speed(gmm, "grouped_mm", 0.45, 0.8)
    v["P4"] = {}
    for k, c in (("auto", auto), ("padded", pad), ("grouped_mm", gmm)):
        v["P4"][k] = parity(ref, c) if (ref and c and ref.get("status") == "ok" and c.get("status") == "ok") else {"verdict": "NOT_READ"}
    if auto and pad and auto.get("status") == "ok" and pad.get("status") == "ok" and auto.get("peak_vram_gb"):
        r = pad["peak_vram_gb"] / auto["peak_vram_gb"]
        v["P5"] = {"ratio": round(r, 4), "verdict": "HOLDS" if r <= 1.4 else "REFUTED"}
    else:
        v["P5"] = {"verdict": "NOT_READ"}
    return out


def render_md(o: dict) -> str:
    L = ["# P46 — the adapter path at the field recipe (Qwen3-30B-A3B, RTX 5090)", "",
         "| arm | status | s/step (med 11+) | tok/s | peak GB | held-out final | path census (per step) |", "|---|---|---|---|---|---|---|"]
    for k, a in o["arms"].items():
        pc = a.get("path_census") or {}
        per = pc.get("per_step", {})
        census = ", ".join(f"{kk.replace('lora_path_', '')}={sorted(set(vv))}" for kk, vv in per.items()) or "—"
        L.append(f"| `{k}` | {a.get('status')} | {a.get('s_per_step_median_11plus')} | {a.get('tokens_per_s')} | {a.get('peak_vram_gb')} | {a.get('eval_loss_final')} | {census} |")
    L += ["", "## Verdicts", ""] + [f"- **{k}**: {json.dumps(x)}" for k, x in o["verdicts"].items()] + [""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    o = reduce(a.run_dir)
    md = render_md(o)
    print(md)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(o, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
