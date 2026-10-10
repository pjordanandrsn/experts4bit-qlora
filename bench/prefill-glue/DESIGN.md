# Prefill glue: a fused path for norm calls above 64 rows (design note, zero rental)

**Status:** item 1 of the maintainer's proposed queue (#846, lane 2). This is a design for review only: no package code
and no rental until it is reviewed. It follows `bench/decode-census/PREFILL.md`, which reads about a quarter of e4b's
512-token prefill forward (40.3 ms on an RTX 5090) as PyTorch eager glue, and every millisecond of that forward as
0.25 ms of TPOT at C = 64. Every number below is computed by `glue_map.py` from P119's committed
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

## 2. The proposal: `E4B_FUSE_PREFILL_GLUE`, default `0`, in three separable parts

- **P1, norms and rotary above 64 rows.** With the knob on, the three hand-back gates let prefill rows through to
  kernels that already exist and take any row count.
  - The input and final norm go to `rmsnorm_rows`, grid `(R,)` (gnf4 `int4_b32.py:1148-1180`).
  - The post-attention norm, with the residual add before it, goes to `rmsnorm_resid_rows` (1184-1235).
  - The q/k norms and the rotary go to `rope_norm_qk`, grid `(R, HQ + HK)` (1314-1364).
  - None of these kernels asserts or caps R.
- **P2, the MoE dispatch, bit-identical by construction.** K19's `gather_div=` reads token rows directly, and
  `scatter=order` writes in the caller's row order (gnf4 `int4_smallm.py:254-257`). Today both sit behind the
  decode-only lean path (`hot_residency.py:646-666`).
- **P3, the router epilogue above 64 rows** (`router_epilogue`, grid `(R,)`, with the bf16 cast in its store).
  - Its expert set is exact.
  - Its top-k slot order can differ from `torch.topk` (P70), and `combine_rows` sums in slot order, so P3 is not
    bitwise. It rides with P1's quality check or is left out.

**What each part would absorb** (P119's 512-token profile, device ms; † means the kernel name is truncated in the receipt
and its time is split between operations by launches per layer):

| group (source operations) | part | device ms | launches |
|---|---|---:|---:|
| RMSNorm chains: 4 a layer + the final norm (fp32 upcast, pow, mean, +eps, rsqrt, mul, downcast, ×weight) † | P1 | 3.72 | 1550 |
| rotary on q and k (`rotate_half`: neg, cat; ×cos, ×sin, add) † | P1 | 1.61 | 480 |
| residual adds (2 a layer; P1 folds the post-attention one) † | P1, half | 0.27 | 96 |
| router epilogue (softmax, top-k, sum, divide, bf16 cast) † | P3 | 0.76 | 293 |
| MoE dispatch: the 4096-row gather of x and the unsort `index_copy_` | P2 | 1.34 | 97 |
| MoE chained tile builder (argsort, scatter_add, cumsums, searchsorted, index_selects, where, int casts) † | out of scope | 3.64 | 2021 |
| attention staging and output copies (`ctx.stage` cats, `.contiguous()`) † | out of scope | 0.99 | 432 |
| per-forward setup (rotary cos/sin tables, masks) and the fp8 KV write at flush | out of scope | 0.33 | 55 |
| already fused in prefill (`_combine_rows`, which P119's table classes as routing) | out of scope | 0.20 | 48 |
| unassigned | | 0.00 | 0 |
| **total (P119's elementwise + routing + device copies)** | | **12.86** | |

- **P1:** about 5.47 ms. **P2:** 1.34 ms. **P3:** 0.76 ms. Together 7.56 ms of the forward's
  12.86 ms of glue.
- **At C = 64,** that is about 1.35 ms of TPOT from P1 alone, or 1.86 ms
  from all three. That is before the fused kernels' own time (not read; a norm over 512 × 2048 bf16 moves about 4 MB).
- **Out of scope here:**
  - the chained tile builder (3.64 ms), which needs a one-launch builder above 1,024 rows: a gnf4 kernel
    item;
  - the attention staging copies (0.99 ms).

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
| **B (recommended)** | the decode kernels above | the most: norm, residual and rotary fused | teacher-forced quality |
| A | `rope_train` + `rmsnorm_train` (HF order) | less: no residual or rotary-norm fusion | bitwise (rotary), ≤ 2 ulp (norm) |

**Why B.** It makes prefill use the arithmetic decode already uses on `qwen3_moe`. That arithmetic is licensed by P115's
read at T == 1 (COMBINED_SANE: bias +0.00541, agreement 0.9674) through `FUSION_DEFAULT_FAMILIES`. So the K/V a
prompt writes and the K/V decode writes come from one function, as `router_epilogue.CAST_WEIGHTS` already arranges for
router weights. **A** is the fallback if B misses its bar.

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

**On one rented RTX A2000 (sm_86, correctness only, no timing quoted).**
- **The forward runs without the paged runner.** Its fp8 paged KV needs sm_89, so the forward is the plain model
  forward, with e4b's folds and the K19 experts.
- **P2, bitwise:**
  - logits at all 512 positions and every layer's hidden state are `torch.equal` with the knob off, over 16 wikitext
    windows;
  - a blindness arm, an `order` with one swapped pair, must differ.
  - This is P127's licence pattern.
- **P1 (and P3), teacher-forced:**
  - P115's instrument scores the prefill forward's own logits at every position: 48 windows × 128 positions on
    wikitext and c4val1, one 512-token chunk.
  - The bar is P115's, against the `chunk` floor (prompts in 256-token chunks): mean `d_ON` ≤ `B_floor` + 0.01 nats,
    `mean|d_ON|` ≤ 2 × max(`S_floor`, 0.005), wikitext K8 `|Δppl|` ≤ 0.05, and SANE (`|mean d|` ≤ 0.02, argmax
    agreement ≥ 0.95).
  - A `mutant_scale` arm must fail.
- **The served path** (fp8 paged KV, the prefill graph's startup bitwise check) is exercised only on an sm_89+ card. That
  would come with a speed lane, after the queue is confirmed.

## 5. Questions for the maintainer

1. Option B (decode arithmetic) or A (HF order)?
2. Should P2 have its own knob? It is bit-identical, so after its bitwise check it could default on alone.
3. Is P3 in or out? It is the smallest part and the only one whose slot order differs.
