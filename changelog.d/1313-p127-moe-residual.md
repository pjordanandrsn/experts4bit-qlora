### The decoder layer's MoE residual add moves into the experts' combine, where it is licensed (#1313, lane P127's Phase 2, item c)

- **What.** With grouped-nf4-gemm's `combine_rows(..., residual=)` (grouped-nf4-gemm#527), the glue fold's decoder layer
  runs its sparse-MoE block's own composition: the gate, then the experts with the layer's residual handed down. On
  the all-resident collapse the residual goes into the combine's epilogue, one launch fewer per layer. Every other
  experts path adds it as the layer did.
- **Licence.** `glue_r2.license_moe_residual` licenses each folded layer, one row count at a time.
  - It probes the model as served: `serve_paged.build_engine` calls it after the residency, the collapse and the
    batched grouping are set, and before any graph is captured.
  - It probes at every decode bucket.
  - A layer is licensed at T rows only where the composition is bitwise the layer's own `residual + mlp(h)` on
    distinct random bf16 inputs.
  - A block with any other body or child is refused, and an unprobed row count keeps the layer's own body.
  - `info["moe_residual"]` reports it.
- **Residuals the kernel would refuse.** The kernel is handed only a bf16 `[T, H]` residual on its own device, and only
  when it takes `residual=`; anything else is the torch add.
- **Knob.** It shares the layer fold's knob, `E4B_FUSE_T1_GLUE_R2`.
- **No speed claim.** P127's read prices it with the rest of Phase 1 and Phase 2 under the bitwise gate.
