# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane 2 design note (zero rental): where e4b's batched decode step spends its time at 16, 32 and 64 rows.

Every number in bench/decode-census/README.md is computed here from committed receipts; nothing is measured anew.

    python bench/decode-census/decode_census.py            # print the note
    python bench/decode-census/decode_census.py --write    # rewrite README.md
    python bench/decode-census/decode_census.py --check    # exit 1 if README.md differs (tests/test_decode_census.py)
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
B = REPO / "bench"
SRC = {
    "p119": B / "p119/receipts/p119-5090-1/box.json",
    "p124": B / "p124/receipts/p124-5090-1/box.json",
    "p126": B / "p126/receipts/p126-5090-1/box.json",
    "sc2e": B / "h2h-2026-10-02/sc2e/receipts/sc2e-5090-1/sc2/verdict_sc2e.json",
    "sc5": B / "sc5/receipts/sc5-5090-2/sc5.json",
    "sc5v": B / "sc5/receipts/sc5-5090-2/verdict.json",
}
CLASSES = [("expert_int4", "int4 experts (K19 grouped GEMM)"), ("attn", "attention (fp8 paged decode, append)"),
           ("dense_int4", "int4 attention projections"), ("moe_route", "routing and tile table"),
           ("elementwise", "elementwise"), ("dense_gemm", "dense GEMM (output head)"), ("norm_glue", "norms and glue")]
OUT_TOKENS = 256          # SC5_MAXTOK: every SC5 request decodes 256 tokens (SC2's 512-token prompt pool)


def _load(key):
    return json.loads(SRC[key].read_text(encoding="utf-8"))


def _classes(cls: dict) -> dict:
    named = {k: cls.get(k, [0, 0.0])[1] for k, _ in CLASSES}
    named["other"] = sum(v[1] for k, v in cls.items() if k not in named)
    return named


def gather() -> dict:
    p119, p124, p126, sc2e, sc5 = (_load(k) for k in ("p119", "p124", "p126", "sc2e", "sc5"))
    cols = {
        16: {"src": "P119 `d16`", "cls": _classes(p119["decode"]["d16"]["classes"]),
             "served": [sc2e["census"][f"s16_d{d}"]["steps"]["decode_steps"]["16"]["step_ms_p50"] for d in (1, 2)],
             "served_src": "SC2e `s16`, draws 1 and 2"},
        32: {"src": "P124 `on32`", "cls": _classes(p124["profile"]["on32"]["classes"]),
             "served": [p124["served"]["32"][a]["median_ms"] for a in ("ON_a", "ON_b")], "served_src": "P124 ON, runners a and b"},
        64: {"src": "P126 attempt 1, P = 4", "cls": _classes(p126["profile"]["4"]["classes"]),
             "served": [p126["served"]["4"][b]["arms"]["4"]["median_ms"] for b in ("a", "b")],
             "served_src": "P126 attempt 1, P = 4, blocks a and b"},
    }
    for c in cols.values():
        c["eager"] = sum(c["cls"].values())
    s16 = sc2e["census"]["s16_d1"]["steps"]["decode_steps"]["16"]
    host16 = s16["step_ms_p50"] - s16["device_ms_p50"]
    stall = st.median(sc2e["census"][a]["steps"]["prefill_steps"]["direct_stall_ms_p50"]
                      for a in ("s16_d1", "s16_d2", "s64a_d1", "s64a_d2"))

    def tpot(fw, mem, c):
        return st.median(b["cells"][str(c)]["tpot_p50_s"] * 1000 for d in ("1", "2") for b in sc5["draws"][d][mem]
                         if b.get("framework") == fw)

    def toks(fw, mem, c):
        return st.median(b["cells"][str(c)]["output_tok_s"] for d in ("1", "2") for b in sc5["draws"][d][mem]
                         if b.get("framework") == fw)
    split = []
    for c, mem, rows in ((16, "default", 16), (16, "matched", 16), (64, "matched", 64)):
        step = st.mean(cols[rows]["served"])
        stall_tok = (c - 1) * stall / OUT_TOKENS
        t = tpot("e4b", mem, c)
        split.append({"c": c, "mem": mem, "tpot": t, "step": step, "stall": stall_tok, "resid": t - step - stall_tok,
                      "vllm": tpot("vllm", mem, c), "e4b_toks": toks("e4b", mem, c), "vllm_toks": toks("vllm", mem, c)})
    lab = _load("sc5v")["labels"]
    ttft = [lab[f"{m}|{c}|{q}|vllm"] for m in ("default", "matched") for c in (1, 16, 64) for q in ("ttft_p50_s", "ttft_p95_s")]
    tpot_t = sorted({f"C = {c} {m}" for m in ("default", "matched") for c in (1, 16, 64) for q in ("tpot_p50_s", "tpot_p95_s")
                     if lab[f"{m}|{c}|{q}|vllm"] == "TRAILS"})
    return {"cols": cols, "host16": host16, "stall": stall, "split": split,
            "ttft_leads": sum(x == "LEADS" for x in ttft), "ttft_n": len(ttft), "tpot_trails": tpot_t}


def render(g: dict) -> str:
    cols, sp = g["cols"], g["split"]
    f = "{:.2f}".format
    rows = ["| class (device ms per step, eager profile) | 16 rows | 32 rows | 64 rows |", "|---|---:|---:|---:|"]
    for k, name in CLASSES + [("other", "other (sampling, copies, misc)")]:
        rows.append(f"| {name} | " + " | ".join(f(cols[r]["cls"][k]) for r in (16, 32, 64)) + " |")
    rows.append("| **eager device total** | " + " | ".join(f"**{f(cols[r]['eager'])}**" for r in (16, 32, 64)) + " |")
    rows.append("| served (captured) step, median | " + " | ".join(" / ".join(f(x) for x in cols[r]["served"]) for r in (16, 32, 64)) + " |")
    rows.append("| **residual** = served − eager total | " +
                " | ".join(" / ".join(f(x - cols[r]["eager"]) for x in cols[r]["served"]) for r in (16, 32, 64)) + " |")
    share64 = {k: cols[64]["cls"][k] / cols[64]["eager"] for k, _ in CLASSES}
    marg = (cols[64]["eager"] - cols[16]["eager"]) / 48
    marg_k19 = (cols[64]["cls"]["expert_int4"] - cols[16]["cls"]["expert_int4"]) / 48
    srows = ["| SC5 cell | e4b TPOT p50 | served decode step | prefill stall per token | **residual** | vLLM TPOT p50 |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in sp:
        srows.append(f"| C = {s['c']}, {s['mem']} | {f(s['tpot'])} | {f(s['step'])} | {f(s['stall'])} | **{s['resid']:+.2f}** | {f(s['vllm'])} |")
    m64 = sp[2]
    work_e4b, work_vllm = 64 / m64["e4b_toks"] * 1000, 64 / m64["vllm_toks"] * 1000
    return f"""# Lane 2 design note: where e4b's batched decode step spends its time at 16, 32 and 64 rows

**Zero rental; for review before any registration** (#846, lane 2 of the serving-throughput program). Nothing here is
measured anew. `decode_census.py` computes every number in sections 1–3 from committed receipts, and
`tests/test_decode_census.py` keeps this file equal to its output. The appendix cites SGLang's source and the reading's
server logs.

**Sources.**
- SC5's reading `sc5-5090-2`: e4b 0.52.0 against vLLM 0.31.0, TPOT per cell.
- SC2e's step traces (`verdict_sc2e.json`): the served 16-row step and the prefill stall.
- Three eager kernel-class profiles:
  - P119 at 16 rows;
  - P124 at 32 rows, with int4 attention projections on (the 0.51.0 default);
  - P126 attempt 1 at 64 rows, with the four-program tile table (the 0.52.0 default).
- Each column comes from its own RTX 5090. Ratios inside a column are within one box; columns are not.

## 1. The decode step, by kernel class

{chr(10).join(rows)}

**Reading the table.**
- The residual closes each column. It is host time and the difference between a captured step at a growing context and
  an eager profile of a few steps.
- At 16 rows, SC2e's own traces put the host part at {f(g['host16'])} ms: step minus device, draw 1.
- At 64 rows, int4 experts are {share64['expert_int4']:.0%} of the eager step, attention {share64['attn']:.0%} and int4
  projections {share64['dense_int4']:.0%}.
- From 16 to 64 rows the eager step grows {marg:.3f} ms a row, of which the expert GEMM is {marg_k19:.3f}.
- At 32 rows the tile table is P124's one-program table. 0.52.0's `E4B_INT4_TILE_PROGRAMS=auto` covers that size too,
  but no lane has read it there.

## 2. SC5's TPOT, split

Closed-loop TPOT is the served decode step plus the prefill stall that every admitted request puts on the decoders.
- SC2e measured that stall at {f(g['stall'])} ms per admitted 512-token prompt: the median of four servers' direct stall.
- Over a request's {OUT_TOKENS} tokens, the other C − 1 slots each admit one request, so the stall per token is
  (C − 1) × {f(g['stall'])} / {OUT_TOKENS} ms.

{chr(10).join(srows)}

**What the split shows.**
- The residual closes each row within about 0.2 ms.
- At C = 64, the prefill stall is {m64['stall'] / m64['tpot']:.0%} of e4b's TPOT. e4b's 64-row decode step
  ({f(m64['step'])} ms) is below vLLM's whole TPOT ({f(m64['vllm'])} ms).
- **Inference, not measured.** Per 64 tokens, e4b's served work is {f(work_e4b)} ms (64 ÷ its output tok/s) against
  vLLM's {f(work_vllm)} ms.
  - If vLLM's 64-row decode step were e4b's, vLLM would spend about {f(work_vllm - m64['step'])} ms per 64 tokens on
    admissions, against e4b's {f(m64['stall'])}.
  - That would make e4b's 512-token prefill forward roughly {m64['stall'] / max(work_vllm - m64['step'], 1e-9):.1f}× vLLM's.
  - No vLLM step-level or prefill-level data exists at these sizes. Only SC1b's 16-row class census does, on a 400 W
    card. A vLLM step census at 64 rows, or a prefill timing at 512 tokens, would settle it.
- **The trade-off SC5 shows** is consistent with this. Against vLLM, e4b leads TTFT in {g['ttft_leads']} of
  {g['ttft_n']} cells and trails TPOT at {", ".join(g['tpot_trails'])}.

## 3. What this suggests for lane 2 (for review; nothing is registered)

1. **The prefill forward per admitted request.** It is about {f(m64['stall'])} ms of every 64-row TPOT and largest at
   high concurrency. The first zero-rental step would be the same class census for the 512-token prefill forward (40 ms
   on SC2e's servers), to see whether its experts, attention or glue dominate.
2. **The int4 expert GEMM (K19)** is {f(cols[64]['cls']['expert_int4'])} ms of the 64-row step and
   {f(cols[16]['cls']['expert_int4'])} ms of the 16-row step. Its byte-floor efficiency at 32 and 64 rows is not read:
   that needs the distinct experts per step at those sizes.
3. **Attention and projections** together are {f(cols[64]['cls']['attn'] + cols[64]['cls']['dense_int4'])} ms at 64
   rows.

## Appendix: SGLang 0.5.21's decode CUDA-graph cap (zero rental)

SC5's read calls SGLang's C = 64 TPOT (48–51 ms, against about 9 ms at C = 16) not diagnosed. The reading's own logs
and SGLang's source account for it.
- **The source.** In the locked wheel (`sglang-0.5.21-cp312-cp312-manylinux_2_34_x86_64.whl`, sha256 `ac300998…`),
  `sglang/srt/arg_groups/memory_hook.py` lines 100–116 set the default decode graph `max_bs` to 48 on a card with 20 to
  35 GiB at tensor parallel below 4. Its comment names the RTX 5090.
- **The box.** The server info reports `cuda_graph_config.decode.max_bs` 48, with graphs captured for batch sizes 1–48.
- **The logs.** In blocks 3 and 9 of the reading (default and matched), every decode batch of 16 or fewer ran with the
  graph (397 log lines each). Every batch above 48 ran without it (32 each). These are the per-block server logs, kept
  with the private receipts.
- At C = 64, SGLang therefore decodes eagerly at its default. This changes no SC5 label; the read's cells stand as
  measured.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(gather())
    readme = Path(__file__).resolve().parent / "README.md"
    if a.write:
        readme.write_bytes(text.encode("utf-8"))
        return 0
    if a.check:
        same = readme.is_file() and readme.read_bytes().decode("utf-8") == text
        print("DECODE_CENSUS", "same" if same else "DIFFERS")
        return 0 if same else 1
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
