### `E4B_ATTN_INT4_WIDE=1`: attention projections of 17–64 rows on the int4 small-M GEMM instead of a cached bf16 copy (opt-in; #846)

- **What changes, opt-in.** With `E4B_ATTN_INT4_WIDE=1`, an `Int4Linear` serves 17–64 rows with grouped-nf4-gemm's
  K16 small-M GEMM, on its own int4 bytes, at a 32- or 64-row tile. That covers the attention projections of a
  batched decode step above 16 rows. Today those rows take a cached bf16 copy and cuBLAS. One row still takes the
  GEMV, 2–16 rows still take K16 exactly as before, and more than 64 rows (prefill) still take the bf16 matmul.
- **Why.** In lane P119 that cuBLAS path cost 2.05 ms of Qwen3-30B-A3B's 15.64 ms 64-row step. It reads 1.81 GB a
  step; the int4 grid it is dequantised from is 510 MB. The speed and the quality are lane P124's to read; nothing
  is claimed here.
- **Needs.** The K16 route on (`E4B_ATTN_INT4_SMALLM` not `0`) and a grouped-nf4-gemm whose `gemm_int4_b32_smallm`
  takes `block_m=` (grouped-nf4-gemm #522). This is detected from the signature, never from a version string. `1` is
  refused at enable time when either is missing, and any value other than `0` or `1` is refused.
- **Memory.** The route's split-K workspace is shared by every projection of one width. It is built zeroed when the
  module is constructed, so it is never born inside a graph capture: there, its zeroing would be recorded into one
  graph rather than run, and another graph sharing it could replay first. A lookup that misses under a capture raises.
  The route runs on one stream at a time, as e4b's runner uses it, so the projections never overlap. That is one
  `4 × 64 × N` fp32 buffer per width (5.2 + 2.1 MB on Qwen3-30B-A3B), not one per projection. While the route is
  opt-in the serve estimate does not price it. `int4_attn.wide_workspace_bytes()` reports it.
- `tests/test_int4_attn_wide.py` covers the route, the sharing and the refusals on the CPU. Its two CUDA tests passed
  on an RTX A2000 (correctness only):
  - two projections of one width captured in one graph replay to their eager bits and within one bf16 ulp of the
    dequant reference;
  - two graphs on the route, captured at 32 then 64 rows, replay to their eager bits with the second replayed first.
