### Added

- `backend-adapter/2`, the adapter manifest a server can verify against (`experts4bit_qlora/adapter_schema.py`;
  `docs/ADAPTER_PORTABILITY.md` section 6). It is defined for `backend="experts4bit"` only.
- `quant_fingerprint(model, scope)` hashes the quantized base in its canonical form, what dequantize computes (scheme
  `qfp-v1`, pinned in the doc). The same base fingerprints alike whether its NF4 absmax is stored nested or expanded,
  while a double-quantized expert absmax fingerprints differently from an fp32 one. The attention scope hashes every
  projection in full. The expert scope hashes every absmax in full plus a fixed byte sample, as a recipe check.
- `lora_scale_applied(model)` reads the scale the LoRA modules hold.
- `prepare_qlora_training`'s report now carries the training facts the writer records, after `enable_fast_train`
  resolves them: the expert absmax form, the fused-q/k/v count, the attention quantization as built, the versions and
  the applied scale.
- The paged server's verifier reads `/2` directories: the v1 checks, every v2 field, a known fingerprint scheme, and
  the applied scale against alpha/r. It refuses a `/2` manifest from another backend by name. A `/2` adapter's expert
  base stays unverified (refusal 9, same opt-in) until the server compares fingerprints (PA-1).
