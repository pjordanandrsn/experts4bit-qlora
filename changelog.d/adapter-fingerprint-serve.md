### Changed

- The paged server now VERIFIES a backend-adapter/2 adapter's base instead of reporting it unverified. After the load and
  before the attention merge, `adapter_dir.fingerprint_check` compares each qfp-v1 fingerprint the manifest records with
  the served base's: the attention projections as loaded, and the expert stacks read from the arena the server serves
  (`quant_fingerprint(..., source="arena")`, PA-1; byte-identical to training's on a real `bake_nf4` arena, tested).
  A match reads `base_match.experts = verified`; `/health` carries both fingerprints per scope.
- An expert fingerprint mismatch is **refusal 12**. `E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH=experts` serves it anyway
  (reported in `/health`); `E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE` does not cover a measured mismatch. An attention
  fingerprint mismatch is **refusal 11**, with no override. v1 adapters are unchanged (refusal 9).
- **Behavior change for an upgrade from 0.57.0:** 0.57.0 served a backend-adapter/2 adapter's expert base as
  "unverified" under `E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts`. It is now MEASURED. An adapter whose base
  matches the arena serves as verified with no opt-in. One whose base does not match (an arena baked from another
  checkpoint or recipe) is refusal 12, which that opt-in no longer covers; set
  `E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH=experts` to serve it anyway, reported in `/health`.
- A double-quantized-absmax adapter always carries expert LoRA (the recipe compresses only under `grouped_nf4`, which
  requires `train_experts=true`), so the paged server refuses it with refusal 1 before the fingerprint check, as
  0.57.0 did. The dq-vs-fp32 branch of refusal 12 applies to a `/2` manifest with zero expert LoRA and to the
  expert-serving path to come.
