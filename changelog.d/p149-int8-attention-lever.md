### `E4B_SERVE_ATTN_INT8=1` (lane P149, Phase 0b): int8 weight-only attention projections through grouped-nf4-gemm's K37 kernels (opt-in)

A new opt-in lever for the paged server stores each attention projection as int8 codes plus fp32 per-row scales, on
grouped-nf4-gemm's per-row grid (`s = absmax / 127`; P147 ruled I8ATTN GO on that grid for Mixtral-8x7B). The bf16
weight is rounded once at the swap, and no bf16 copy is kept.
- A call with 1 row runs K37's int8 GEMV. A call with 2 to 16 rows runs K37's small-M kernel. Both go through
  `dense_i8.linear_i8`.
- A call with more than 16 rows (a prefill), and any CPU or non-bf16 call, dequantizes transiently to `bf16(q * s)` and
  runs `F.linear`. That is P147's fake-quant round trip, bit for bit.
- On Mixtral the census is exactly 4 per layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`: 128 at 32 layers). It is checked
  before anything is swapped, and the output head is never touched. The swapped modules keep their names, so R2's rope
  fold still patches every attention module; the build asserts that after the fusions.
- `/health`'s `attn_int8` records the census and, per row count, which projections routed through K37's kernels or
  dequantized during the build's graph captures and warm-ups. `levers_env` carries the flag.
- Refused:
  - a grouped-nf4-gemm without `dense_i8` (K37);
  - a value other than `0` or `1`;
  - combining it with int4 attention (`E4B_SERVE_ATTN_INT4` or `E4B_SERVE_ATTN_INT4_CALIB`) or with
    `E4B_DENSE_GEMV_BW=1`;
  - a biased or non-bf16 projection;
  - a census miss;
  - `fuse_qkv` on int8 projections (Qwen3-MoE's fused attention is not built for them).

The default is off, and nothing changes unless it is set. Its quality and speed are P149's box Q+S, read after a
grouped-nf4-gemm release carries K37.
