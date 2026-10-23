### Changed

- The paged server now VERIFIES a backend-adapter/2 adapter's base instead of reporting it unverified. After the load and
  before the attention merge, `adapter_dir.fingerprint_check` compares each qfp-v1 fingerprint the manifest records with
  the served base's: the attention projections as loaded, and the expert stacks read from the arena the server serves
  (`quant_fingerprint(..., source="arena")`, PA-1; byte-identical to training's on a real `bake_nf4` arena, tested).
  A match reads `base_match.experts = verified`; `/health` carries both fingerprints per scope.
- **Behavior change for an upgrade from 0.57.0:** a backend-adapter/2 adapter trained with the default double-quantized
  expert absmax was served with `E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts` as "unverified". The server now
  MEASURES the mismatch with the arena's fp32 absmax and refuses it as **refusal 12**, which that opt-in does not cover.
  Set `E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH=experts` to serve it anyway (reported in `/health`), or retrain at
  `E4B_ABSMAX_DQ=0`; a follow-up records the fp32 source fingerprint at training, which verifies the base (base
  identity only, not the served absmax form). An attention
  fingerprint mismatch is **refusal 11**, with no override. v1 adapters are unchanged (refusal 9).
