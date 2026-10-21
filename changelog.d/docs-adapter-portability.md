### Added

- `docs/ADAPTER_PORTABILITY.md`, a compatibility matrix read from code: for each way an adapter is trained (bf16 or
  `attn_4bit` attention, expert absmax fp32 or double-quantized) and each way the paged server can serve the base
  (bf16, merged into a rebuilt `Q_train(W)`, the attention levers, the NF4 arena, int4 or MXFP4 stores), it states
  whether the served function is exact by construction, bounded-loss behind a registered gate, or refused, with code
  references. It finds that today's default training (double-quantized expert absmax) is not exact on today's NF4
  arena (fp32 absmax), and that the adapter manifest does not yet record what an exact rebuild needs. No numbers are
  measured here.
