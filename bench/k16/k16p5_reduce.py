#!/usr/bin/env python3
"""bench/k16/k16p5_reduce.py -- read K16's P5 from the three-arm census (nf4_b16 / int4_b16 / int4_b16_smallm).

    python bench/k16/k16p5_reduce.py <run-dir> [--md RESULTS-k16p5.md] [--json out.json]

P5 (grouped-nf4-gemm kernel/PREREG-k16-smallm-int4-gemm.md): routed into Int4Linear for 1 < R <= 16, "P42's census on
the same box shows the attention GEMM row falling by >= 0.4 ms/step at B=16". Read here as: the bf16 GEMM kernels'
ms/step (the cached-bf16 matmul the attention projections take at R=16 when the route is off) fall by >= 0.4 ms/step
from `int4_b16` to `int4_b16_smallm`, with the K16 kernel's own ms/step in the smallm arm recorded beside it, and the
timed `step_ms_clean` of both arms quoted. The bf16 GEMM family is matched by name (`--gemm-regex`); the top kernels
of every arm are printed so the match can be checked by eye -- a regex that swallowed an unrelated kernel would show
up there. P42's parser is reused (bench/p42/p42_reduce.py: ms/step = self-CUDA over the profiled replays).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("p42_reduce", Path(__file__).resolve().parents[1] / "p42" / "p42_reduce.py")
p42 = importlib.util.module_from_spec(_spec)
sys.modules["p42_reduce"] = p42
_spec.loader.exec_module(p42)

ARMS = ["nf4_b16", "int4_b16", "int4_b16_smallm"]
K16_KERNEL = "gemm_int4_b32_smallm"
DEFAULT_GEMM_REGEX = r"gemm|cutlass|cublas|xmma|nvjet|Sm90|sm80|sm90"
P5_MS = 0.4


def arm_census(d: Path, arm: str):
    f = d / "logs" / f"census_{arm}.txt"
    if not f.exists():
        return None
    replays, rows = p42.parse_census(f.read_text())
    for r in rows:
        r["ms_per_step"] = r["self_ms"] / max(replays, 1)
    return {"replays": replays, "rows": rows}


def step_ms(d: Path, arm: str):
    try:
        return p42.step_ms(d, arm)
    except Exception:
        f = d / f"e4b_b16_{arm}.json"
        if not f.exists():
            return None
        j = json.load(open(f))
        return j.get("step_ms_clean") or j.get("step_ms")


def reduce(run_dir: str, gemm_regex: str = DEFAULT_GEMM_REGEX) -> dict:
    d = Path(run_dir)
    rx = re.compile(gemm_regex, re.I)
    out = {"arms": {}, "gemm_regex": gemm_regex}
    for arm in ARMS:
        c = arm_census(d, arm)
        s = step_ms(d, arm)
        if c is None:
            out["arms"][arm] = {"census": "MISSING", "step_ms_clean": s}
            continue
        gemm = [r for r in c["rows"] if rx.search(r["name"]) and K16_KERNEL not in r["name"]]
        k16 = [r for r in c["rows"] if K16_KERNEL in r["name"]]
        top = sorted(c["rows"], key=lambda r: -r["ms_per_step"])[:12]
        out["arms"][arm] = {
            "step_ms_clean": s, "replays": c["replays"], "n_kernels": len(c["rows"]),
            "bf16_gemm_ms_per_step": sum(r["ms_per_step"] for r in gemm), "bf16_gemm_kernels": len(gemm),
            "k16_ms_per_step": sum(r["ms_per_step"] for r in k16), "k16_calls_per_step": sum(r["calls"] for r in k16) / max(c["replays"], 1),
            "top": [{"name": r["name"][:80], "ms_per_step": round(r["ms_per_step"], 4)} for r in top],
        }
    a, b = out["arms"].get("int4_b16", {}), out["arms"].get("int4_b16_smallm", {})
    if "bf16_gemm_ms_per_step" in a and "bf16_gemm_ms_per_step" in b:
        drop = a["bf16_gemm_ms_per_step"] - b["bf16_gemm_ms_per_step"]
        engaged = b["k16_ms_per_step"] > 0 and a["k16_ms_per_step"] == 0
        out["p5"] = {"bf16_gemm_drop_ms_per_step": drop, "k16_ms_per_step_in_smallm_arm": b["k16_ms_per_step"],
                     "net_attention_gemm_change_ms_per_step": drop - b["k16_ms_per_step"],
                     "route_engaged_only_in_smallm_arm": engaged, "threshold_ms_per_step": P5_MS,
                     "step_ms_clean_int4_b16": a.get("step_ms_clean"), "step_ms_clean_int4_b16_smallm": b.get("step_ms_clean"),
                     "verdict": ("P5_HOLDS" if engaged and drop >= P5_MS else "P5_REFUTED" if engaged else "NOT_READ (route not engaged as expected)")}
    else:
        out["p5"] = {"verdict": "NOT_READ", "reason": "an int4_b16 census is missing"}
    return out


def render_md(v: dict) -> str:
    L = ["# K16 P5 — the attention GEMM row with the small-M route on (B=16, RTX 5090)", "",
         f"bf16 GEMM family matched by `/{v['gemm_regex']}/`; K16 kernel `{K16_KERNEL}`. ms/step = self-CUDA over the profiled replays (P42's parser).", "",
         "| arm | step_ms_clean | bf16 GEMM ms/step (kernels) | K16 ms/step (calls/step) | kernels |", "|---|---|---|---|---|"]
    for arm, r in v["arms"].items():
        if r.get("census") == "MISSING":
            L.append(f"| `{arm}` | {r.get('step_ms_clean')} | — | — | census MISSING |")
        else:
            L.append(f"| `{arm}` | {r['step_ms_clean']} | {r['bf16_gemm_ms_per_step']:.3f} ({r['bf16_gemm_kernels']}) | {r['k16_ms_per_step']:.3f} ({r['k16_calls_per_step']:.0f}) | {r['n_kernels']} |")
    p = v["p5"]
    L += ["", f"**P5: {p['verdict']}**" + (f" — bf16 GEMM drop {p['bf16_gemm_drop_ms_per_step']:+.3f} ms/step (threshold ≥ {p['threshold_ms_per_step']}), K16 kernel {p['k16_ms_per_step_in_smallm_arm']:.3f} ms/step in the smallm arm, net attention-GEMM change {p['net_attention_gemm_change_ms_per_step']:+.3f} ms/step; step_ms_clean {p['step_ms_clean_int4_b16']} → {p['step_ms_clean_int4_b16_smallm']}" if "bf16_gemm_drop_ms_per_step" in p else f" — {p.get('reason', '')}"), ""]
    for arm, r in v["arms"].items():
        if r.get("top"):
            L += [f"### top kernels — `{arm}`", "", "| kernel | ms/step |", "|---|---|"] + [f"| `{t['name']}` | {t['ms_per_step']:.4f} |" for t in r["top"]] + [""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gemm-regex", default=DEFAULT_GEMM_REGEX)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    v = reduce(a.run_dir, a.gemm_regex)
    md = render_md(v)
    print(md)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
