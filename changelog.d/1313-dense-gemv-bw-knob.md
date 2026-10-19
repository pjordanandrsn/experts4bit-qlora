### `E4B_DENSE_GEMV_BW=1` (#1313, lane P151): the attention projections' single-row decode GEMVs through grouped-nf4-gemm's K35 kernel (opt-in)

A new opt-in knob for the paged server routes the fused q/k/v and o projections through `dense_gemv.gemv_bf16_bw` when a
call has one row. That is a decode step at one request, or the bucket-1 CUDA graph. The kernel is grouped-nf4-gemm's K35:
experts4bit-qlora P148's prototype GEMV, shipped verbatim under a tolerance contract against cuBLAS.
- The swapped module shares the projection's weight; there is no copy.
- More rows (a prefill, or a decode bucket above 1), CPU tensors and other dtypes go to `F.linear` unchanged.
- Refused: a grouped-nf4-gemm without `dense_gemv`, a projection with a bias or a non-bf16 weight, and a model where
  nothing matched.
- `/health`'s `levers.dense_gemv_bw` records the modules swapped and, per row count, which modules routed through the kernel
  during the build's graph captures: the engagement record a served read checks.

The default is `0`, and nothing changes unless it is set. Its speed is P151's served read, registered after a
grouped-nf4-gemm release carries K35.
