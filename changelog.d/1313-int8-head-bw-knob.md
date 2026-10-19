### `E4B_INT8_HEAD_BW=1` (#1313, lane P153): the lm_head on its int8 grid, its single-row decode GEMV through grouped-nf4-gemm's K36 kernel (opt-in)

A new opt-in knob for the paged server serves the output head on its per-row int8 grid. It routes the head's one-row
calls (a decode step at one request, or the bucket-1 CUDA graph) through `dense_gemv.gemv_i8_bw`. That is
grouped-nf4-gemm's K36: experts4bit-qlora P148's int8 prototype kernel and quantizer, shipped verbatim under a tolerance
contract against cuBLAS on the bf16 head.
- **Knob-on changes multi-row outputs too.** The head's bf16 weight is rounded in place to its int8 grid cast back to
  bf16, exactly P150's I8HEAD round trip. So every row count computes the head P150 ruled GO on quality at both gates,
  and the served model never depends on the batch size.
- One-row calls go to the kernel over the same codes and fp32 scales. A non-contiguous one-row input is copied, never
  passed to `F.linear`. More rows, CPU tensors and other dtypes go to `F.linear` on the rounded weight.
- The head keeps its weight storage (no second bf16 copy) and adds the int8 codes and scales, about 0.31 GB at
  Qwen3-30B-A3B's head.
- Refused: a grouped-nf4-gemm without `gemv_i8_bw`, a head that is not an `nn.Linear` (an int4 head), a head tied to the
  input embeddings, a bias, and a weight that is not bf16.
- `/health`'s `levers.int8_head_bw` records the head swapped and, per row count, whether it routed through the kernel
  or passed during the build's graph captures: the engagement record a served read checks.

The default is `0`, and nothing changes unless it is set. Its speed is P153's served read.
