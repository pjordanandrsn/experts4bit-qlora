### Added

- The paged server serves one trained adapter (`E4B_PAGED_ADAPTER=<dir>`, Loggetta's `backend-adapter/1`
  directory). The new `experts4bit_qlora/adapter_dir.py` verifies the directory before any weight is read: paths,
  sizes, sha256, the tensor set, shapes, dtypes and finiteness, read with safetensors and never pickle. It then merges
  the attention LoRA into the bf16 projections right after the load (`W + (alpha/r)·B·A` in fp32, cast once, with
  the scale from the manifest), before the levers, fusions, GEMV routes and graph capture, so serving the adapter adds
  no per-token work. The merge reports each projection's rounding error.
- Startup refuses, naming every reason, and never serves the base model in the adapter's place:
  - a live expert LoRA;
  - a base revision mismatch, or an arena whose repo or revision differs;
  - an arena quantization recipe that differs or is not recorded;
  - a lever that re-quantizes an adapted module or depends on calibration read on the base model (the uncalibrated,
    weight-only int8 lm_head stays on as a base-serving property);
  - unknown target modules, or a verifier failure;
  - attention trained on a 4-bit base;
  - an expert base a v1 manifest cannot verify;
  - speculative decoding.
- Two of those refusals have explicit opt-ins: `E4B_PAGED_ADAPTER_ALLOW_UNVERIFIED_ARENA=1` and
  `E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts`.
- The served name becomes `<model>+<adapter_id[:12]>`. `/health` and `/v1/models` report the adapter's id, its
  manifest hash, the base revision, the merge report and `base_match`.
