# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane 2 design note (a), zero rental: where e4b's 512-token prefill forward spends its time.

Every number in bench/decode-census/PREFILL.md is computed here from committed receipts; nothing is measured anew.
Kernels are classed by P119's own frozen table (bench/p119/p119_box.py ``CLASSES``), read from that file's source.

    python bench/decode-census/prefill_census.py            # print the note
    python bench/decode-census/prefill_census.py --write    # rewrite PREFILL.md
    python bench/decode-census/prefill_census.py --check    # exit 1 if PREFILL.md differs (tests/test_prefill_census.py)
"""
import argparse
import ast
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
B = REPO / "bench"
SRC = {
    "p119": B / "p119/receipts/p119-5090-1/box.json",
    "p119_box": B / "p119/p119_box.py",
    "sc2e": B / "h2h-2026-10-02/sc2e/receipts/sc2e-5090-1/sc2/verdict_sc2e.json",
    "sc5": B / "sc5/receipts/sc5-5090-2/sc5.json",
}
# Qwen3-30B-A3B config.json (revision ad44e777): the shapes the FLOP counts use
LAYERS, HIDDEN, MOE_INTER, TOP_K, Q_DIM, KV_DIM = 48, 2048, 768, 8, 32 * 128, 4 * 128
TOKENS = 512                      # P119's prefill and SC2's prompt pool
DISTINCT_EXPERTS = 87             # P100: distinct experts per layer in a 512-token prefill chunk
BYTES_PER_PARAM = 4.5 / 8         # int4-b32, 4.50 bpw
HBM_GBPS = 1792                   # RTX 5090 memory bandwidth, GB/s (vendor figure)
SUB = [("copies, casts and cat", ("copy", "cat", "bfloat16")),
       ("index, gather and scatter", ("index", "gather", "scatter")),
       ("binary arithmetic (adds, muls)", ("binaryfunctor", "functor_add", "mul", "add", "div", "sub")),
       ("norm math (eager RMSNorm: mean, pow, rsqrt)", ("meanops", "pow_tensor", "rsqrt")),
       ("softmax, where, fill", ("softmax", "where", "fill")),
       ("other elementwise", ("",))]
SUB_ORDER = ["norm math (eager RMSNorm: mean, pow, rsqrt)", "index, gather and scatter", "copies, casts and cat",
             "softmax, where, fill", "binary arithmetic (adds, muls)", "other elementwise"]


def _load(key):
    return json.loads(SRC[key].read_text(encoding="utf-8"))


def _kclass():
    """P119's frozen ``CLASSES`` table and first-match rule, from its source (the module itself imports torch)."""
    tree = ast.parse(SRC["p119_box"].read_text(encoding="utf-8"))
    table = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CLASSES" for t in n.targets))

    def kclass(name):
        low = str(name).lower()
        return next((cls for cls, keys in table if any(k in low for k in keys)), "other")
    return kclass


def _sub(name):
    low = name.lower()
    for key in SUB_ORDER:
        if any(k in low for k in dict(SUB)[key]):
            return key
    return "other elementwise"


def gather() -> dict:
    p119, sc2e, sc5 = _load("p119"), _load("sc2e"), _load("sc5")
    kclass = _kclass()
    pf = p119["prefill"]["p512_on"]
    classes = {k: v[1] for k, v in pf["classes"].items()}
    sub, sub_calls, route = {}, {}, {}
    for name, calls, ms in pf["kernels"]:
        c = kclass(name)
        if c == "elementwise":
            s = _sub(name)
            sub[s] = sub.get(s, 0.0) + ms
            sub_calls[s] = sub_calls.get(s, 0.0) + calls
        elif c == "moe_route":
            short = "radix sort (`radixSortKVInPlace`)" if "radixSort" in name else "other routing (top-k, scans, combine)"
            route[short] = route.get(short, 0.0) + ms
    proj_ms = sum(ms for name, _c, ms in pf["kernels"] if kclass(name) == "dense_gemm" and "128x128" in name)
    expert_flop = 2 * TOKENS * LAYERS * TOP_K * 3 * HIDDEN * MOE_INTER
    proj_flop = 2 * TOKENS * LAYERS * (HIDDEN * (Q_DIM + 2 * KV_DIM) + Q_DIM * HIDDEN)
    expert_bytes = DISTINCT_EXPERTS * LAYERS * 3 * HIDDEN * MOE_INTER * BYTES_PER_PARAM
    served = {d: sc2e["census"][f"s16_d{d}"]["steps"]["prefill_steps"] for d in (1, 2)}
    ttft1 = st.median(b["cells"]["1"]["ttft_p50_s"] * 1000 for d in ("1", "2") for b in sc5["draws"][d]["default"]
                      if b.get("framework") == "e4b")
    return {"device": pf["device_ms"], "device_off": p119["prefill"]["p512_off"]["device_ms"], "classes": classes,
            "sub": sub, "sub_calls": sub_calls, "route": route, "proj_ms": proj_ms, "expert_flop": expert_flop,
            "proj_flop": proj_flop, "expert_bytes": expert_bytes, "served": served, "ttft1": ttft1,
            "e4b": p119["e4b_sha"][:8], "gnf4": p119["gnf4_sha"][:8]}


def render(g: dict) -> str:
    f = "{:.2f}".format
    tot = sum(g["classes"].values())
    names = {"expert_int4": "int4 experts (K19 grouped GEMM)", "elementwise": "elementwise (PyTorch eager kernels)",
             "dense_gemm": "dense GEMM (bf16 attention projections, router, head)", "moe_route": "routing",
             "attn": "attention (flash)", "other": "other (`_swiglu_rows`, misc)", "memcpy_dtod": "device-to-device copies"}
    rows = ["| class | device ms | share |", "|---|---:|---:|"]
    for k, v in sorted(g["classes"].items(), key=lambda kv: -kv[1]):
        if v >= 0.05:
            rows.append(f"| {names.get(k, k)} | {f(v)} | {v / tot:.1%} |")
    small = sum(v for v in g["classes"].values() if v < 0.05)
    rows.append(f"| copies, sampling (under 0.05 ms each) | {f(small)} | {small / tot:.1%} |")
    rows.append(f"| **total (eager, last-token logits)** | **{f(tot)}** | |")
    srows = ["| elementwise, by kernel name | device ms | launches |", "|---|---:|---:|"]
    for k in [s for s, _ in SUB]:
        srows.append(f"| {k} | {f(g['sub'].get(k, 0))} | {g['sub_calls'].get(k, 0):.0f} |")
    rrows = [f"| {k} | {f(v)} |" for k, v in sorted(g["route"].items(), key=lambda kv: -kv[1])]
    s1, s2 = g["served"][1], g["served"][2]
    exp_tf = g["expert_flop"] / (g["classes"]["expert_int4"] * 1e-3) / 1e12
    proj_tf = g["proj_flop"] / (g["proj_ms"] * 1e-3) / 1e12
    exp_at_dense = g["expert_flop"] / (proj_tf * 1e12) * 1e3
    exp_bw_ms = g["expert_bytes"] / (HBM_GBPS * 1e9) * 1e3
    per_ms_c64 = 63 / 256
    return f"""# Lane 2 design note (a): where e4b's 512-token prefill forward spends its time

**Zero rental; for review before any registration** (#846, lane 2). This follows `README.md` in this directory: at
C = 64, SC5's TPOT carries about 10 ms per token of prefill stall, and each admitted 512-token prompt's forward is the
stall. `prefill_census.py` computes every number below from committed receipts, and `tests/test_prefill_census.py` keeps
this file equal to its output.

**Sources.**
- P119's eager profile of one 512-token prefill: `prefill.p512_on`, with logits for the last token only, as served.
  The stack was e4b `{g['e4b']}` and grouped-nf4-gemm `{g['gnf4']}` on an RTX 5090. Kernels are classed by P119's own
  frozen table.
- SC2e's served prefill steps (`verdict_sc2e.json`, arm `s16`).
- SC5's C = 1 TTFT on 0.52.0.

## 1. The forward, by class

{chr(10).join(rows)}

The elementwise class is PyTorch eager kernels around the fused ones; by kernel name:

{chr(10).join(srows)}

Routing:

| routing, by kernel | device ms |
|---|---:|
{chr(10).join(rrows)}

## 2. Closing it against the served prefill

- **The served forward.** SC2e's served forward, device p50, is {f(s1['forward_device_ms_p50'])} / {f(s2['forward_device_ms_p50'])}
  ms in draws 1 / 2. The residual against this eager total is {s1['forward_device_ms_p50'] - tot:+.2f} /
  {s2['forward_device_ms_p50'] - tot:+.2f} ms, from a different box and graph replay against eager.
- **The served prefill step** is {f(s1['step_ms_p50'])} / {f(s2['step_ms_p50'])} ms. That is the forward plus
  {f(s1['step_ms_p50'] - s1['forward_device_ms_p50'])} / {f(s2['step_ms_p50'] - s2['forward_device_ms_p50'])} ms of
  riding decode ({s1['riding_decode_rows_p50']:.0f} / {s2['riding_decode_rows_p50']:.1f} rows p50) and host. The direct
  stall it puts on the decoders is {f(s1['direct_stall_ms_p50'])} / {f(s2['direct_stall_ms_p50'])} ms.
- **A consistency check, not a split.** SC5's C = 1 TTFT p50 on 0.52.0 is {f(g['ttft1'])} ms: one forward of about
  40 ms plus the first decode and the host.
- **The full-vocabulary logits** P119 also profiled (`p512_off`, {f(g['device_off'])} ms) cost
  {f(g['device_off'] - g['device'])} ms more. Serving keeps the last token only.

## 3. Rates on the same profile

- **The bf16 attention projections** (the two 128×128 CUTLASS kernels, 96 launches: q|k|v and o in 48 layers):
  {g['proj_flop'] / 1e12:.3f} TFLOP in
  {f(g['proj_ms'])} ms, which is **{proj_tf:.0f} TFLOPS**. That is this card's achieved dense bf16 rate, read from the
  same profile.
- **The int4 experts (K19):** {g['expert_flop'] / 1e12:.3f} TFLOP in {f(g['classes']['expert_int4'])} ms, which is
  **{exp_tf:.0f} TFLOPS**, {exp_tf / proj_tf:.0%} of that dense rate. At the dense rate they would take {f(exp_at_dense)} ms.
- **Expert weight bytes:** P100's {DISTINCT_EXPERTS} distinct experts per layer at 4.50 bpw is
  {g['expert_bytes'] / 1e9:.1f} GB, which is {f(exp_bw_ms)} ms at {HBM_GBPS} GB/s.
- So the prefill experts are compute-bound, and run at under half the card's achieved dense rate.

## 4. What this suggests for lane 2 (for review; nothing is registered)

At C = 64, each millisecond off this forward is about {per_ms_c64:.2f} ms of TPOT: (C − 1) / 256 per admitted prompt.
1. **The eager glue:** {f(g['classes']['elementwise'])} ms ({g['classes']['elementwise'] / tot:.0%}) across about
   {sum(g['sub_calls'].values()):.0f} launches, plus the {f(g['route'].get('radix sort (`radixSortKVInPlace`)', 0))} ms
   radix sort in routing.
   - Copies, casts and `cat` are the largest part.
   - The decode path's fused RMSNorm (`experts4bit_qlora/engines/glue_fuse.py`) hands any call over 64 rows back to
     the eager chain by design ("Prefill ... keep the original chain"). The prefill therefore runs PyTorch's norm math.
   - **The first candidate:** it is a code change with a bitwise or teacher-forced check, not a kernel project.
2. **K19 at the prefill shape** (about 32 rows per expert): {f(g['classes']['expert_int4'])} ms, at {exp_tf / proj_tf:.0%}
   of the dense rate. The ceiling on what a prefill-shaped tile configuration could recover is
   {f(g['classes']['expert_int4'] - exp_at_dense)} ms.
3. **The bf16 projections** already run at the card's dense rate. int4 would save bytes, and the compute-bound
   prefill does not need them.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(gather())
    note = Path(__file__).resolve().parent / "PREFILL.md"
    if a.write:
        note.write_bytes(text.encode("utf-8"))
        return 0
    if a.check:
        same = note.is_file() and note.read_bytes().decode("utf-8") == text
        print("PREFILL_CENSUS", "same" if same else "DIFFERS")
        return 0 if same else 1
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
