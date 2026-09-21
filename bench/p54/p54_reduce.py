#!/usr/bin/env python3
"""bench/p54/p54_reduce.py -- read P54 (bench/p54/P54-PREREG.md) from the fetched run directory.

    python bench/p54/p54_reduce.py <run-dir> [--md RESULTS-p54.md] [--json out.json]

Arms: `int4_b16` / `int4_b16_fqkv` / `int4_b1` / `int4_b1_fqkv`, each drawn twice (`<arm>` and `<arm>_r2` receipts;
the census on the first draw only), plus the untimed `int4_b16_series` arm (`series_int4_b16.json.gz`). Reads:

* **P1 / P2** -- the saving in ms/step, fused vs control, from the MEDIAN of each arm's two `step_ms_clean` draws
  (B=16 band 0.25-0.50, refuted under 0.10; B=1 band 0.20-0.45, refuted under 0.10). Both draws are printed so the
  A/A spread is visible beside the effect.
* **P3** -- the mechanism: `_gemm_int4_b32_smallm` calls/step 192 -> 96 at B=16; every other kernel family within
  +-5 % of its control row. Kernel rows are grouped by the same family regexes P42 used; the top rows of every arm
  are printed so a regex that swallowed something shows up by eye.
* **P4** -- the distinct-expert count: mean over layers and steps of the number of distinct experts each layer
  touched per decode step, from the series file (a list per layer of per-step touched-expert id lists).
* **STOP-1** -- the control's first draw within +-8 % of K16 P5's 11.19 ms (a cross-box check that gates nothing but
  the right to compare against that page's numbers).

P42's parser is reused (bench/p42/p42_reduce.py: ms/step = self-CUDA over the profiled replays). Nothing is
recomputed from memory: a missing receipt is a missing row, never a zero.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("p42_reduce", Path(__file__).resolve().parents[1] / "p42" / "p42_reduce.py")
p42 = importlib.util.module_from_spec(_spec)
sys.modules["p42_reduce"] = p42
_spec.loader.exec_module(p42)

PAIRS = {16: ("int4_b16", "int4_b16_fqkv"), 1: ("int4_b1", "int4_b1_fqkv")}
K16_KERNEL = "gemm_int4_b32_smallm"
GEMV_KERNEL = "_gemv_int4_b32"
BANDS = {16: (0.25, 0.50), 1: (0.20, 0.45)}   # prereg P1 / P2
REFUTE_UNDER = 0.10
K16P5_CONTROL_MS = 11.19                       # bench/k16/RESULTS-k16-p5.md, int4_b16_smallm on that box
STOP1_TOL = 0.08
FAMILIES = [
    ("k16 small-M GEMM", re.compile(r"gemm_int4_b32_smallm")),
    ("int4 expert GEMV", re.compile(r"_gemv_int4_b32")),
    ("bf16 GEMM (cutlass/cublas)", re.compile(r"cutlass|cublas|xmma|nvjet|gemvx|Sm90|sm80|sm90", re.I)),
    ("fp8 paged decode", re.compile(r"paged_decode|f8dot")),
    ("index / gather / scatter", re.compile(r"index|gather|scatter|_combine_rows", re.I)),
    ("activation quant", re.compile(r"_quant_x_rows")),
    ("split-K reduce", re.compile(r"_reduce_partials|reduce_kernel")),
    ("rope/norm folds", re.compile(r"rope_norm_heads|rmsnorm")),
]


def arm_census(d: Path, arm: str):
    f = d / "logs" / f"census_{arm}.txt"
    if not f.exists():
        return None
    replays, rows = p42.parse_census(f.read_text())
    for r in rows:
        r["ms_per_step"] = r["self_ms"] / max(replays, 1)
        r["calls_per_step"] = r.get("calls", 0) / max(replays, 1)
    return {"replays": replays, "rows": rows}


def step_ms(d: Path, arm: str, batch: int, suffix: str = ""):
    f = d / f"e4b_b{batch}_{arm}{suffix}.json"
    if not f.exists():
        return None
    try:
        return float(json.loads(f.read_text())["step_ms_clean"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def draws(d: Path, arm: str, batch: int):
    out = [step_ms(d, arm, batch, s) for s in ("", "_r2")]
    have = [x for x in out if x is not None]
    return {"draws": out, "median": statistics.median(have) if have else None, "n": len(have)}


def families(c):
    out = {}
    for name, rx in FAMILIES:
        rows = [r for r in c["rows"] if rx.search(r["name"])]
        out[name] = {"ms_per_step": sum(r["ms_per_step"] for r in rows),
                     "calls_per_step": sum(r["calls_per_step"] for r in rows), "kernels": len(rows)}
    return out


def series_stats(d: Path):
    f = d / "series_int4_b16.json.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        ser = json.load(fh)["per_layer_series"]
    per_layer = []
    for layer in ser:
        counts = [len(set(step)) for step in layer]
        per_layer.append({"steps": len(counts), "mean_distinct": statistics.mean(counts) if counts else None,
                          "min": min(counts) if counts else None, "max": max(counts) if counts else None})
    means = [x["mean_distinct"] for x in per_layer if x["mean_distinct"] is not None]
    return {"layers": len(per_layer), "mean_distinct_over_layers": statistics.mean(means) if means else None,
            "min_layer_mean": min(means) if means else None, "max_layer_mean": max(means) if means else None,
            "per_layer": per_layer}


def reduce(run_dir: str) -> dict:
    d = Path(run_dir)
    out = {"pairs": {}, "series": series_stats(d), "stop1": None}
    for batch, (ctl, fq) in PAIRS.items():
        pair = {"control": draws(d, ctl, batch), "fused": draws(d, fq, batch)}
        cm, fm = pair["control"]["median"], pair["fused"]["median"]
        pair["saving_ms"] = None if (cm is None or fm is None) else cm - fm
        lo, hi = BANDS[batch]
        sv = pair["saving_ms"]
        if sv is None:
            pair["verdict"] = "UNREAD (missing draw)"
        elif sv < REFUTE_UNDER:
            pair["verdict"] = f"REFUTED (saving {sv:.3f} < {REFUTE_UNDER})"
        elif sv < lo:
            pair["verdict"] = f"SMALLER THAN PREDICTED, REAL (saving {sv:.3f} in [{REFUTE_UNDER}, {lo}))"
        elif sv <= hi:
            pair["verdict"] = f"HOLDS (saving {sv:.3f} in band [{lo}, {hi}])"
        else:
            pair["verdict"] = f"OVER THE BAND ({sv:.3f} > {hi}) -- read the census before believing it"
        cc, fc = arm_census(d, ctl), arm_census(d, fq)
        pair["census"] = {}
        if cc and fc:
            cf, ff = families(cc), families(fc)
            moved = {}
            for name in cf:
                a, b = cf[name]["ms_per_step"], ff[name]["ms_per_step"]
                rel = None if a == 0 else (b - a) / a
                moved[name] = {"control_ms": a, "fused_ms": b, "rel": rel,
                               "control_calls": cf[name]["calls_per_step"], "fused_calls": ff[name]["calls_per_step"]}
            pair["census"] = {"families": moved, "replays": (cc["replays"], fc["replays"]),
                              "top_control": sorted(cc["rows"], key=lambda r: -r["ms_per_step"])[:10],
                              "top_fused": sorted(fc["rows"], key=lambda r: -r["ms_per_step"])[:10]}
            if batch == 16:
                k16c = moved["k16 small-M GEMM"]
                pair["p3_calls"] = (k16c["control_calls"], k16c["fused_calls"])
                pair["p3_calls_ok"] = abs(k16c["control_calls"] - 192) < 1 and abs(k16c["fused_calls"] - 96) < 1
            else:
                g = moved["int4 expert GEMV"]
                pair["p3_calls"] = (g["control_calls"], g["fused_calls"])
                pair["p3_calls_ok"] = abs((g["control_calls"] - g["fused_calls"]) - 96) < 1
            others = [n for n, v in moved.items() if n not in ("k16 small-M GEMM", "bf16 GEMM (cutlass/cublas)")
                      and v["rel"] is not None and abs(v["rel"]) > 0.05 and v["control_ms"] > 0.05]
            pair["p3_others_moved_over_5pct"] = others
        else:
            pair["census"] = {"MISSING": [n for n, c in ((ctl, cc), (fq, fc)) if c is None]}
        out["pairs"][batch] = pair
    c16 = out["pairs"][16]["control"]["draws"][0] if 16 in out["pairs"] else None
    if c16 is not None:
        out["stop1"] = {"control_first_draw_ms": c16, "k16p5_ms": K16P5_CONTROL_MS,
                        "rel": c16 / K16P5_CONTROL_MS - 1, "within": abs(c16 / K16P5_CONTROL_MS - 1) <= STOP1_TOL}
    return out


def _fmt(v):
    return "--" if v is None else f"{v:.3f}"


def to_md(rep: dict) -> str:
    L = ["# Results -- P54: qkv fusion on the int4 attention store", "",
         "Pre-registration: [`P54-PREREG.md`](P54-PREREG.md). Every number below is read from the fetched receipts by `p54_reduce.py`.", ""]
    s1 = rep["stop1"]
    if s1:
        L += [f"**STOP-1** -- control first draw {s1['control_first_draw_ms']:.3f} ms vs K16 P5's {s1['k16p5_ms']} ms on its box: "
              f"{s1['rel']*100:+.1f} % ({'within' if s1['within'] else 'OUTSIDE'} +-8 %; ratios within this box are the position either way).", ""]
    for batch in (16, 1):
        pair = rep["pairs"][batch]
        L += [f"## B={batch}: `{PAIRS[batch][0]}` vs `{PAIRS[batch][1]}`", "",
              "| arm | draw 1 (ms) | draw 2 (ms) | median |", "|---|---|---|---|"]
        for tag in ("control", "fused"):
            dr = pair[tag]["draws"]
            L.append(f"| {tag} | {_fmt(dr[0])} | {_fmt(dr[1])} | {_fmt(pair[tag]['median'])} |")
        saving = "--" if pair["saving_ms"] is None else f"{pair['saving_ms']:.3f} ms/step"
        L += ["", f"**Saving (control median - fused median): {saving}** -> **{pair['verdict']}**", ""]
        cen = pair["census"]
        if "families" in cen:
            L += ["| kernel family | control ms/step | fused ms/step | rel | control calls/step | fused calls/step |", "|---|---|---|---|---|---|"]
            for name, v in cen["families"].items():
                rel = "--" if v["rel"] is None else f"{v['rel']*100:+.1f} %"
                L.append(f"| {name} | {v['control_ms']:.3f} | {v['fused_ms']:.3f} | {rel} | {v['control_calls']:.0f} | {v['fused_calls']:.0f} |")
            L += ["", f"P3 calls (control, fused): {pair.get('p3_calls')} -> {'OK' if pair.get('p3_calls_ok') else 'NOT as predicted'}; "
                  f"other families moved > 5 %: {pair.get('p3_others_moved_over_5pct') or 'none'}", ""]
            for tag in ("top_control", "top_fused"):
                L += [f"<details><summary>{tag} (top 10 kernels)</summary>", "", "| kernel | ms/step | calls/step |", "|---|---|---|"]
                L += [f"| `{r['name'][:90]}` | {r['ms_per_step']:.3f} | {r['calls_per_step']:.0f} |" for r in cen[tag]]
                L += ["", "</details>", ""]
        else:
            L += [f"Census: {cen}", ""]
    ser = rep["series"]
    L += ["## P4: distinct experts per layer per decode step (B=16, `int4_b16_series`, untimed)", ""]
    if ser and ser["mean_distinct_over_layers"] is not None:
        L += [f"Mean over {ser['layers']} layers: **{ser['mean_distinct_over_layers']:.1f}** distinct experts per layer per step "
              f"(layer means {ser['min_layer_mean']:.1f} .. {ser['max_layer_mean']:.1f}; uniform-random expectation 80.9; prereg prior <= 80). "
              "Read against #564's expert-tier roofline: 64 distinct = 5.336 ms, 80 = 6.670 ms, measured GEMV ~6.2-6.7 ms.", ""]
    else:
        L += ["Series file MISSING -- P4 unread.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    rep = reduce(a.run_dir)
    md = to_md(rep)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    print(md)


if __name__ == "__main__":
    main()
