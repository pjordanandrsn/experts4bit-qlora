### Added

- A default-trained (double-quantized expert absmax) backend-adapter/2 adapter now VERIFIES its expert base on the
  paged server. `compress_expert_absmax_` hashes each stack's full fp32 absmax before compressing it, and
  `quant_fingerprint(model, "experts")` carries `fp32_source` (the base's qfp-v1 digest with those fp32 absmax, plus
  the hash per stack). The server compares it with its fp32 arena exactly: equal reads `base_match.experts = verified
  (source, ...); trained on double-quantized absmax, served on fp32 absmax` and serves without an override; different
  is refusal 12 naming the first stack that differs. It proves the base is the trained one, not that the served
  absmax equals the trained one: the quality effect of serving the fp32 form is not read. A /2 manifest without the
  field is refusal 12 as before. The writer needs no change (it records `quant_fingerprint`'s dict as returned); the
  hash adds one host copy and sha256 of the expert absmax at compression (about 1.8 GB on Qwen3-30B-A3B).
