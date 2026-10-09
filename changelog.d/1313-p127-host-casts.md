### Three host casts gone from the T == 1 decode (#1313, lane P127's Phase 1)

- **What.** Three host-side casts are dropped, each one a kernel launch per MoE layer. The kernels already do the
  conversion, so the outputs are unchanged:
  - The fused router forwards (`router_epilogue.py`, all three kinds) hand `router_epilogue` the logits in the
    projection's own dtype. The kernel loads them with `.to(tl.float32)`, so the host `logits.float()` was redundant.
  - The collapsed hot path's combine (`hot_residency._combine_topk`) hands `combine_rows` the top-k weights in the
    router's dtype. The kernel widens them on load. The torch fallback chain still casts its weights to fp32.
  - The NF4 grouped route casts the expert ids to int32 once, for both its `gemm_4bit_grouped` calls. Before, the
    wrapper cast them on each call. The int4 and MXFP4 routes already did this.
- **Why it is value-preserving.** bf16 -> fp32 widening is exact, the ids are the same integers, and every
  grouped-nf4-gemm since e4b's floor (0.30.0) widens on load in both kernels. Tests:
  - CPU tests assert what each kernel is handed and that the result is bitwise what the cast gave. Each is
    mutation-checked: restoring the cast fails it.
  - A CUDA test runs the real kernels both ways and asserts identical outputs.
- **No speed claim.** P127's read prices these together with Phase 2 (gnf4 kernel options) under a bitwise gate.
