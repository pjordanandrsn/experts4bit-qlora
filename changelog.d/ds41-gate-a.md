### Added

- `bench/ds41-gate-a/`: a metadata-only architecture audit of DeepSeek-V4.1-Flash-1017 (revision `1bae0e10`; no weights
  downloaded) and a support matrix for e4b and gnf4. The text backbone is 551.88B parameters (763.21B with the Engram tables,
  vision and MTP). Its MXFP4 experts match e4b's and gnf4's byte layout. The model type `deepseek_v41` has no module tree in
  transformers, so e4b cannot load it today. Documentation and one metadata script.
