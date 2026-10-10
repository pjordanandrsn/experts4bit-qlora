# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Design note (zero rental): a fused path for norm calls above 64 rows. The glue numbers are computed here.

Every kernel of P119's eager 512-token prefill profile (``prefill.p512_on``) that P119's frozen table classes as
elementwise, routing or a device copy is assigned to the source operation that launches it, read off the prefill call
chain. The receipt stores kernel names truncated at 160 characters, so a kernel whose functor is cut off and that several
operations share is split by its launches per layer; DESIGN.md marks those with a dagger.

    python bench/prefill-glue/glue_map.py --write     # rewrite DESIGN.md
    python bench/prefill-glue/glue_map.py --check     # exit 1 if DESIGN.md differs (tests/test_prefill_glue_map.py)
"""
import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P119 = REPO / "bench/p119/receipts/p119-5090-1/box.json"
P119_BOX = REPO / "bench/p119/p119_box.py"
LAYERS = 48
C64_TPOT_PER_MS = 63 / 256      # bench/decode-census/PREFILL.md: each ms of forward is (C - 1) / 256 ms of TPOT at C = 64

# group -> (label, part of the proposal or None when out of scope)
GROUPS = {
    "norm": ("RMSNorm chains: 4 a layer + the final norm (fp32 upcast, pow, mean, +eps, rsqrt, mul, downcast, ×weight)", "P1"),
    "rope": ("rotary on q and k (`rotate_half`: neg, cat; ×cos, ×sin, add)", "P1"),
    "resid": ("residual adds (2 a layer; P1 folds the post-attention one)", "P1, half"),
    "router": ("router epilogue (softmax, top-k, sum, divide, bf16 cast)", "P3"),
    "dispatch": ("MoE dispatch: the 4096-row gather of x and the unsort `index_copy_`", "P2"),
    "tiles": ("MoE chained tile builder (argsort, scatter_add, cumsums, searchsorted, index_selects, where, int casts)", None),
    "attn_copies": ("attention staging and output copies (`ctx.stage` cats, `.contiguous()`)", None),
    "setup": ("per-forward setup (rotary cos/sin tables, masks) and the fp8 KV write at flush", None),
    "fused": ("already fused in prefill (`_combine_rows`, which P119's table classes as routing)", None),
}
# (substring(s) that must all appear, launches per layer or None, {group: fraction}, dagger) -- first match wins
RULES = [
    (("MeanOps",), None, {"norm": 1}, False),
    (("pow_tensor_scalar",), None, {"norm": 1}, False),
    (("rsqrt_kernel",), None, {"norm": 1}, False),
    (("CUDAFunctorOnSelf_add<float>",), None, {"norm": 1}, False),
    (("BinaryFunctor<float, float, float", "MulFu"), None, {"norm": 1}, False),
    (("unrolled_elementwise_kernel<at::native::direct_copy", "lambda()#7"), None, {"norm": 1}, False),
    (("bfloat16_copy_kernel",), None, {"norm": 4 / 5, "router": 1 / 5}, True),
    (("BinaryFunctor<c10::BFloat16",), None, {"norm": 1 / 2, "rope": 1 / 2}, True),
    (("neg_kernel",), None, {"rope": 1}, False),
    (("CatArrayBatchedCopy<",), 2.0, {"rope": 1}, True),
    (("CUDAFunctor_add<c10::BFloat16>",), None, {"rope": 1 / 2, "resid": 1 / 2}, True),
    (("softmax_warp_forward",), None, {"router": 1}, False),
    (("gatherTopK",), None, {"router": 1}, False),
    (("bitonicSortKVInPlace",), None, {"router": 1}, False),
    (("sum_functor",), None, {"router": 1}, False),
    (("BinaryFunctor<float, float, float", "DivFu"), None, {"router": 1}, False),
    (("_combine_rows",), None, {"fused": 1}, False),
    (("vectorized_gather_kernel",), None, {"dispatch": 1}, False),
    (("index_copy_kernel_impl<at::native::OpaqueType<2>",), None, {"dispatch": 1}, False),
    (("radixSortKVInPlace",), None, {"tiles": 1}, False),
    (("DeviceScan",), None, {"tiles": 1}, False),
    (("searchsorted",), None, {"tiles": 1}, False),
    (("_scatter_gather_elementwise",), None, {"tiles": 1}, False),
    (("indexSelect",), None, {"tiles": 1}, False),
    (("<long>",), None, {"tiles": 1}, False),
    (("<long,",), None, {"tiles": 1}, False),
    (("launch_clamp",), None, {"tiles": 1}, False),
    (("where_kernel_impl",), 3.0, {"tiles": 1}, False),
    (("CompareFunctor",), None, {"tiles": 1}, False),
    (("BitwiseAnd",), None, {"tiles": 1}, False),
    (("arange_cuda_out",), None, {"tiles": 1}, False),
    (("unrolled_elementwise_kernel<at::native::direct_copy", "lambda()#3"), 3.0, {"tiles": 1}, True),
    (("direct_copy_kernel_cuda",), 2.0, {"attn_copies": 1}, True),
    (("direct_copy_kernel_cuda",), 3.0, {"attn_copies": 1}, True),
    (("Memcpy DtoD",), None, {"attn_copies": 1}, False),
]


def _kclass():
    tree = ast.parse(P119_BOX.read_text(encoding="utf-8"))
    table = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CLASSES" for t in n.targets))
    return lambda name: next((c for c, keys in table if any(k in str(name).lower() for k in keys)), "other")


def assign(name: str, per_layer: float) -> tuple:
    """(fractions by group, dagger) for one kernel; per-forward kernels (well under one launch a layer) are setup."""
    if per_layer < 0.5:
        return {"setup": 1}, False
    for subs, count, frac, dagger in RULES:
        if all(s in name for s in subs) and (count is None or abs(per_layer - count) < 0.15):
            return frac, dagger
    return {}, False


def gather() -> dict:
    kclass = _kclass()
    pf = json.loads(P119.read_text(encoding="utf-8"))["prefill"]["p512_on"]
    ms, launches, dag, unassigned = {g: 0.0 for g in GROUPS}, {g: 0.0 for g in GROUPS}, {g: False for g in GROUPS}, []
    scope = 0.0
    for name, calls, t in pf["kernels"]:
        if kclass(name) not in ("elementwise", "moe_route", "memcpy_dtod"):
            continue
        scope += t
        frac, dagger = assign(name, calls / LAYERS)
        if not frac:
            unassigned.append((name, calls, t))
        for g, f in frac.items():
            ms[g] += t * f
            launches[g] += calls * f
            dag[g] = dag[g] or dagger
    cls = {k: v[1] for k, v in pf["classes"].items()}
    return {"ms": ms, "launches": launches, "dagger": dag, "unassigned": unassigned, "scope": scope,
            "device": pf["device_ms"], "elementwise": cls["elementwise"], "route": cls["moe_route"],
            "dtod": cls.get("memcpy_dtod", 0.0)}


def render(g: dict) -> str:
    f = "{:.2f}".format
    rows = ["| group (source operations) | part | device ms | launches |", "|---|---|---:|---:|"]
    for k, (label, part) in GROUPS.items():
        rows.append(f"| {label}{' †' if g['dagger'][k] else ''} | {part or 'out of scope'} | {f(g['ms'][k])} | {g['launches'][k]:.0f} |")
    un = sum(t for _n, _c, t in g["unassigned"])
    rows.append(f"| unassigned | | {f(un)} | {sum(c for _n, c, _t in g['unassigned']):.0f} |")
    rows.append(f"| **total (P119's elementwise + routing + device copies)** | | **{f(g['scope'])}** | |")
    p1 = g["ms"]["norm"] + g["ms"]["rope"] + g["ms"]["resid"] / 2
    p2, p3 = g["ms"]["dispatch"], g["ms"]["router"]
    return f"""# Prefill glue: a fused path for norm calls above 64 rows (design note, zero rental)

**Status:** item 1 of the maintainer's proposed queue (#846, lane 2). This is a design for review only: no package code
and no rental until it is reviewed. It follows `bench/decode-census/PREFILL.md`, which reads about a quarter of e4b's
512-token prefill forward (40.3 ms on an RTX 5090) as PyTorch eager glue, and every millisecond of that forward as
{C64_TPOT_PER_MS:.2f} ms of TPOT at C = 64. Every number below is computed by `glue_map.py` from P119's committed
profile, and `tests/test_prefill_glue_map.py` keeps this file equal to its output.

## 1. What runs today in a 512-token prefill (one layer, Qwen3-30B-A3B, 0.52.0 defaults)

Read from e4b `bfb0a682`, grouped-nf4-gemm `bf6184f` and transformers 5.16.1.

- **The decoder layer.** glue_r2's layer fold hands any call above 64 rows back to HF's
  `Qwen3MoeDecoderLayer.forward` (`engines/glue_r2.py:264-270`; `_MAX_DECODE_ROWS = 64`, 53-54).
- **Every RMSNorm** (input, post-attention, q_norm, k_norm, and the final norm once) is HF's eager chain:
  - `glue_fuse._fwd` returns any call above 64 rows to it (`engines/glue_fuse.py:178-185`), with the comment "Prefill and
    exotic dtypes keep the original chain";
  - HF's chain is `to(float32)`, `pow(2).mean`, `rsqrt(var + eps)`, a multiply, `to(bf16)`, `weight *`.
  - That is about 8 launches a norm, and the counts match the profile exactly: 193 norms × 3 = 579 pow/mean/rsqrt
    launches.
- **The rotary** is HF's `apply_rotary_pos_emb` (via `engines/qkv_fuse.py:50-66`): `rotate_half`'s `cat((-x2, x1))`,
  then `q·cos + rotate_half(q)·sin`, and the same for k. glue_r2's attention fold hands it back above 64 rows
  (`glue_r2.py:407-415`).
- **The two residual adds** are HF's (`modeling_qwen3_moe.py:341, 347`).
- **The router** is HF's softmax, `topk`, sum, divide and cast. The router-epilogue fold hands it back above 64 rows
  (`engines/router_epilogue.py:409-420`).
- **The MoE dispatch** is e4b code:
  - the gather `x_t.index_select(0, row_token)` expands 4,096 routed rows (`hot_residency.py:688`);
  - the chained tile builder runs because the one-launch builder refuses more than 1,024 rows (gnf4
    `int4_b32.py:961-965`; `nf4_grouped.py:830-854`, about 30 launches a layer);
  - the unsort is `out.index_copy_(0, order, dn)` (`hot_residency.py:1119-1120`).
- **Already fused at every row count:** the q|k|v projection, SwiGLU and the top-k combine.

## 2. The proposal: two knobs, both default `0`: `E4B_FUSE_PREFILL_GLUE` (P1) and `E4B_PREFILL_LEAN_DISPATCH` (P2)

- **P1, norms and rotary above 64 rows.** With the knob on, the three hand-back gates let prefill rows through to
  kernels that already exist and take any row count.
  - The input and final norm go to `rmsnorm_rows`, grid `(R,)` (gnf4 `int4_b32.py:1148-1180`).
  - The post-attention norm, with the residual add before it, goes to `rmsnorm_resid_rows` (1184-1235).
  - The q/k norms and the rotary go to `rope_norm_qk`, grid `(R, HQ + HK)` (1314-1364).
  - None of these kernels asserts or caps R.
- **P2, the MoE dispatch, bit-identical by construction.** K19's `gather_div=` reads token rows directly, and
  `scatter=order` writes in the caller's row order (gnf4 `int4_smallm.py:254-257`). Today both sit behind the
  decode-only lean path (`hot_residency.py:646-666`).
  - The equivalence holds on the same kernel and tile table, so P2's check asserts the route rather than assuming it
    (section 4).
  - The knob is P2's own. Turning it on by default is a later PR, after the A2000 read passes.
- **P3, the router epilogue above 64 rows** (`router_epilogue`, grid `(R,)`, with the bf16 cast in its store).
  - Its expert set is exact.
  - Its top-k slot order can differ from `torch.topk` (P70), and `combine_rows` sums in slot order, so P3 is not
    bitwise.
  - **Out of this round** (maintainer, 2026-10-10). Folded into P1's quality read, it would leave a P1 failure
    unattributable. It can return later as its own arm.

**What each part would absorb** (P119's 512-token profile, device ms; † means the kernel name is truncated in the receipt
and its time is split between operations by launches per layer):

{chr(10).join(rows)}

- **P1:** about {f(p1)} ms. **P2:** {f(p2)} ms. Together {f(p1 + p2)} ms of the forward's {f(g['scope'])} ms of glue.
  P3, out of this round, is {f(p3)} ms.
- **At C = 64,** that is **at most** {f(p1 * C64_TPOT_PER_MS)} ms of TPOT from P1 alone, and **at most**
  {f((p1 + p2) * C64_TPOT_PER_MS)} ms with P2.
  - These are ceilings, not savings: the fused kernels' own time is not subtracted. It is not read; a norm over
    512 × 2048 bf16 moves about 4 MB.
- **Out of scope here:**
  - the chained tile builder ({f(g['ms']['tiles'])} ms), which needs a one-launch builder above 1,024 rows: a gnf4 kernel
    item;
  - the attention staging copies ({f(g['ms']['attn_copies'])} ms).

## 3. The arithmetic question, which decides the check

HF's RMSNorm rounds twice: `round(w · round(x · rsqrt(var + eps)))`. The gnf4 kernels round once: they compute
`x · (1 / sqrt(ms + eps)) · w` in fp32 and store bf16, with a different row-sum order (`int4_b32.py:1163-1167`).
- `rope_norm_qk` likewise replaces HF's chain of roundings with one.
- **None of them is bitwise with HF,** and no test compares them with HF's composite. Their tests compare against an
  fp32-weight reference at a relative bar of 2^-7.
- **e4b has two training kernels that follow HF's order** (wired only by `enable_fast_train`):
  - `engines/rope_train.py` is bit-identical to HF's rotary on CUDA (`tests/test_rope_train.py`, `torch.equal`);
  - `engines/rmsnorm_train.py` with `MUL_FP32=False` follows HF's two roundings, within 2 ulp because its row reduction
    runs in another order.

| option | kernels | what P1 saves | the check P1 needs |
|---|---|---|---|
| **B (decided)** | the decode kernels above | the most: norm, residual and rotary fused | teacher-forced quality |
| A | `rope_train` + `rmsnorm_train` (HF order) | less: no residual or rotary-norm fusion | bitwise (rotary), ≤ 2 ulp (norm) |

**Why B (decided by the maintainer, 2026-10-10).** Prefill then writes K/V with the same function decode uses on
`qwen3_moe`, as `router_epilogue.CAST_WEIGHTS` already arranges for router weights.
- That arithmetic is licensed for decode by P115's read at T == 1 (COMBINED_SANE: bias +0.00541, agreement 0.9674).
- That read does not license it for prefill. The teacher-forced prefill read in section 4 is B's own licence.
- **A** is the fallback if B misses its bar.

## 4. The checks I would register (correctness only: CPU, then one A2000)

**On CPU, no rental.**
- **gnf4, under `TRITON_INTERPRET=1`** (the `interp-contract` job):
  - `rmsnorm_rows`, `rmsnorm_resid_rows` and `rope_norm_qk` at prefill shapes (R = 512, and 4,096 head rows),
    against the existing references at the existing bars;
  - the bitwise tests skip under the interpreter, whose bf16 cast does not round to nearest.
- **e4b, with the torch stubs these tests already use:**
  - With the knob off, every existing hand-back assertion holds unchanged: `tests/test_p115_quality_box.py:184-185`,
    `test_glue_fuse.py:53-56`, `test_glue_r2.py:147, 625, 744`, `test_router_epilogue.py:116, 246`.
  - With the knob on, each fold engages above 64 rows.
  - `tests/test_p115_quality_box.py`'s tiny Qwen3-MoE gains a prefill-shaped arm.

**On the project's own RTX A2000** (sm_86; no rental, $0; correctness only, no timing quoted).
- **The forward runs without the paged runner.** Its fp8 paged KV needs sm_89, so the forward is the plain model
  forward, with e4b's folds and the K19 experts.
- **P2, bitwise:**
  - logits at all 512 positions and every layer's hidden state are `torch.equal` with the knob off, over 16 wikitext
    windows;
  - a blindness arm, an `order` with one swapped pair, must differ;
  - **the route is asserted, not assumed.** On both knob arms the read records, and asserts, that every 512-row expert
    call goes through K19 over the chained 16-row tile table, the route P119 profiled. If the plain forward lands on
    another GEMM or grouping, the read says nothing about P2 and is VOID.
  - This is P127's licence pattern.
- **P1, teacher-forced.** P3 is out of this round, so a P1 failure stays attributable:
  - P115's instrument scores the prefill forward's own logits at every position: 48 windows × 128 positions on
    wikitext and c4val1, one 512-token chunk.
  - The bar is P115's, against the `chunk` floor (prompts in 256-token chunks): mean `d_ON` ≤ `B_floor` + 0.01 nats,
    `mean|d_ON|` ≤ 2 × max(`S_floor`, 0.005), wikitext K8 `|Δppl|` ≤ 0.05, and SANE (`|mean d|` ≤ 0.02, argmax
    agreement ≥ 0.95).
  - A `mutant_scale` arm must fail.
- **The served path** (fp8 paged KV, the prefill graph's startup bitwise check) is exercised only on an sm_89+ card. That
  would come with a speed lane, after the queue is confirmed.

## 5. Decided (maintainer, 2026-10-10)

1. **Option B.** Prefill and decode write K/V with one function. The teacher-forced prefill read is B's licence; A is
   the fallback.
2. **P2 has its own knob,** default 0 in the implementation PR. Turning it on is a later PR, after the A2000
   `torch.equal` read and its blindness arm pass, with the route asserted on both arms.
3. **P3 is out of this round.** It can return later as its own arm.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(gather())
    doc = Path(__file__).resolve().parent / "DESIGN.md"
    if a.write:
        doc.write_bytes(text.encode("utf-8"))
        return 0
    if a.check:
        same = doc.is_file() and doc.read_bytes().decode("utf-8") == text
        print("PREFILL_GLUE", "same" if same else "DIFFERS")
        return 0 if same else 1
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
