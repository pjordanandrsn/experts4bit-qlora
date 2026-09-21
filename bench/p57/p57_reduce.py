#!/usr/bin/env python3
"""bench/p57/p57_reduce.py -- read P57 (bench/p57/P57-PREREG.md) from the fetched run directory.

    python bench/p57/p57_reduce.py <run-dir> [--md RESULTS-p57.md] [--json out.json]

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

PAIRS = {"b1_fr": ("int4_b1", "int4_b1_fr", 1, "split-K reduce"),
         "b16_fr": ("int4_b16", "int4_b16_fr", 16, "split-K reduce"),
         "b16_nor2": ("int4_b16_nor2", "int4_b16_fqkv_nor2", 16, "k16 small-M GEMM")}
K16_KERNEL = "gemm_int4_b32_smallm"
GEMV_KERNEL = "_gemv_int4_b32"
BANDS = {"b1_fr": (0.15, 0.40), "b16_fr": (0.10, 0.35), "b16_nor2": (0.0, 9.9)}   # prereg P1 / P2; the nor2 pair is a parity read, not a saving
REFUTE_UNDER = {"b1_fr": 0.08, "b16_fr": 0.05, "b16_nor2": -1.0}
P54_CONTROL_MS = {"b1_fr": 4.210, "b16_fr": 11.420, "b16_nor2": 11.420}   # bench/p54/RESULTS-p54.md medians on that box
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


def distinct_stats(d: Path):
    """The P57 hook's report (bench/p57/distinct_experts.py): per-layer decode-step series of distinct experts."""
    f = d / "distinct_experts_b16.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def token_parity(d: Path, arm_a: str, arm_b: str, batch: int, suffix_a: str = "", suffix_b: str = ""):
    """P5, read from the receipts rather than by eye: the generated token ids of two arms, per sequence.

    `step_decomp` records `tokens` in every decode receipt (a dict of per-sequence lists at B>1, one list at
    B=1). Two arms of the same configuration on the same box are bit-deterministic -- the control-vs-control
    pair is the A/A that proves it -- so a divergence between control and fused is a real arithmetic
    difference, and its FIRST index per sequence is the honest way to report it.
    """
    def toks(arm, suffix):
        f = d / f"e4b_b{batch}_{arm}{suffix}.json"
        if not f.exists():
            return None
        t = json.loads(f.read_text()).get("tokens")
        if t is None:
            return None
        return {str(k): list(v) for k, v in t.items()} if isinstance(t, dict) else {"0": list(t)}
    A, B = toks(arm_a, suffix_a), toks(arm_b, suffix_b)
    if A is None or B is None or set(A) != set(B):
        return None
    rows = []
    for k in sorted(A, key=int):
        x, y = A[k], B[k]
        first = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        rows.append({"seq": k, "len": (len(x), len(y)), "first_divergence": first})
    identical = [r for r in rows if r["first_divergence"] is None and r["len"][0] == r["len"][1]]
    return {"sequences": len(rows), "identical": len(identical),
            "first_divergences": [(r["seq"], r["first_divergence"]) for r in rows if r["first_divergence"] is not None],
            "all_identical": len(identical) == len(rows)}


def reduce(run_dir: str) -> dict:
    d = Path(run_dir)
    out = {"pairs": {}, "stop1": None}
    for label, (ctl, fq, batch, target) in PAIRS.items():
        pair = {"control": draws(d, ctl, batch), "fused": draws(d, fq, batch), "batch": batch, "arms": (ctl, fq)}
        cm, fm = pair["control"]["median"], pair["fused"]["median"]
        pair["saving_ms"] = None if (cm is None or fm is None) else cm - fm
        lo, hi = BANDS[label]
        sv = pair["saving_ms"]
        refute_under = REFUTE_UNDER[label]
        if sv is None:
            pair["verdict"] = "UNREAD (missing draw)"
        elif label == "b16_nor2":
            pair["verdict"] = f"parity read (saving {sv:.3f} ms is incidental: the split asks about TOKENS, see P3)"
        elif sv < refute_under:
            pair["verdict"] = f"REFUTED (saving {sv:.3f} < {refute_under})"
        elif sv < lo:
            pair["verdict"] = f"SMALLER THAN PREDICTED, REAL (saving {sv:.3f} in [{refute_under}, {lo}))"
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
            if target == "split-K reduce":
                # K17 route: the fr arm carries NO _reduce_partials row; the control does
                red_c = sum(r["calls_per_step"] for r in cc["rows"] if "_reduce_partials" in r["name"])
                red_f = sum(r["calls_per_step"] for r in fc["rows"] if "_reduce_partials" in r["name"])
                pair["p3_calls"] = (red_c, red_f)
                pair["p3_calls_ok"] = red_c > 0 and red_f == 0
            else:
                k16c = moved["k16 small-M GEMM"]
                pair["p3_calls"] = (k16c["control_calls"], k16c["fused_calls"])
                pair["p3_calls_ok"] = abs(k16c["control_calls"] - 192) < 1 and abs(k16c["fused_calls"] - 96) < 1
            # The TARGET family is the one the change removes launches from: the split-K reduce for the K17
            # pairs (the fused epilogue retires _reduce_partials), the K16 GEMM for the round-2 split.
            pair["p3_target_family"] = target
            others = [n for n, v in moved.items() if n not in (target, "bf16 GEMM (cutlass/cublas)")
                      and v["rel"] is not None and abs(v["rel"]) > 0.05 and v["control_ms"] > 0.05]
            pair["p3_others_moved_over_5pct"] = others
        else:
            pair["census"] = {"MISSING": [n for n, c in ((ctl, cc), (fq, fc)) if c is None]}
        # P5, mechanically: the fused arm's generated tokens against the control's on BOTH draws, with the
        # control's own two draws as the A/A that says whether a divergence is arithmetic or run-to-run noise.
        pair["tokens"] = {
            "draw1_control_vs_fused": token_parity(d, ctl, fq, batch),
            "draw2_control_vs_fused": token_parity(d, ctl, fq, batch, "_r2", "_r2"),
            "aa_control_draw1_vs_draw2": token_parity(d, ctl, ctl, batch, "", "_r2"),
        }
        out["pairs"][label] = pair
    out["stop1"] = {}
    for label in ("b1_fr", "b16_fr"):
        c = out["pairs"][label]["control"]["draws"][0] if label in out["pairs"] else None
        if c is not None:
            ref = P54_CONTROL_MS[label]
            out["stop1"][label] = {"control_first_draw_ms": c, "p54_ms": ref, "rel": c / ref - 1, "within": abs(c / ref - 1) <= STOP1_TOL}
    out["distinct"] = distinct_stats(d)
    return out


def _fmt(v):
    return "--" if v is None else f"{v:.3f}"


def to_md(rep: dict) -> str:
    L = ["# Results -- P57: K17's fused reduce in the consumer, P54's B=16 divergence split, the distinct-expert count", "",
         "Pre-registration: [`P57-PREREG.md`](P57-PREREG.md). Every number below is read from the fetched receipts by `p57_reduce.py`.", ""]
    for label, s1 in (rep.get("stop1") or {}).items():
        L += [f"**STOP-1 ({label})** -- control first draw {s1['control_first_draw_ms']:.3f} ms vs P54's {s1['p54_ms']} ms on its box: "
              f"{s1['rel']*100:+.1f} % ({'within' if s1['within'] else 'OUTSIDE'} +-8 %; ratios within this box are the position either way).", ""]
    for label in ("b1_fr", "b16_fr", "b16_nor2"):
        pair = rep["pairs"][label]
        ctl, fq, batch, _t = PAIRS[label]
        L += [f"## {label}: `{ctl}` vs `{fq}` (B={batch})", "",
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
        tk = pair.get("tokens") or {}
        L += ["**P5 (token parity)** -- generated tokens, fused vs control:", ""]
        for label, key in (("draw 1", "draw1_control_vs_fused"), ("draw 2", "draw2_control_vs_fused"),
                           ("A/A (control draw 1 vs draw 2)", "aa_control_draw1_vs_draw2")):
            v = tk.get(key)
            if v is None:
                L.append(f"- {label}: unread (a receipt is missing)")
            elif v["all_identical"]:
                L.append(f"- {label}: **identical** on all {v['sequences']} sequence(s)")
            else:
                L.append(f"- {label}: **{v['sequences'] - v['identical']} of {v['sequences']} sequences diverge**; "
                         f"first divergence (seq, index): {v['first_divergences'][:8]}")
        L.append("")
    dist = rep.get("distinct")
    L += ["## P4: distinct experts per layer per decode step (B=16, `int4_b16_distinct`, untimed)", ""]
    if dist and dist.get("mean_distinct_over_layers") is not None:
        L += [f"Mean over {dist['layers']} layers x {dist['steps']} decode steps: **{dist['mean_distinct_over_layers']:.1f}** distinct experts per layer "
              f"(layer means {dist['min_layer_mean']:.1f} .. {dist['max_layer_mean']:.1f}; uniform-router expectation {dist['uniform_random_expectation']:.1f}; "
              f"prereg prior <= 80). Against #564's expert-tier roofline (5.336 ms at 64 distinct, 6.670 at 80) this sizes the headroom of the largest row in the step.", ""]
    else:
        L += ["`distinct_experts_b16.json` MISSING -- P4 unread.", ""]
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
