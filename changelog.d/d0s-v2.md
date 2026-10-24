### D0 served read: a derived backend-adapter/2 manifest for D0's v1 fixtures, and a shared absmax installer

- `bench/d0s/d0s_v2.py` (CPU, USD 0) writes a NEW adapter directory for a backend-adapter/1 fixture. It holds byte
  copies of every file the v1 manifest lists, so the adapter id is unchanged, and a /2 manifest whose added fields come
  from evidence:
  - the training receipt's recorded `engaged` block;
  - e4b's `quant_fingerprint` over the STORED trained base, after Loggetta's `_expert_digest` over it equals the
    trainer's recorded digest;
  - qfp-v1 over the checkpoint's bf16 attention.

  The scale is "equal by the reader rule; the training code applies alpha/r", checked by reading `lora.py` at the
  training commit. Every link that does not verify refuses. Two runs give byte-identical manifests. `derived_from`
  records every input's sha256 and the limitation: the fingerprints were measured after training, over the verified
  stored base, not recorded by the trainer, and 2 of 16 stacks are verified against the trainer digest.
- `absmax_dq.install_compressed_absmax_(mod, payloads)` installs a GIVEN double-quantized payload exactly as
  `compress_expert_absmax_` leaves a stack, which now installs through it. A stored payload is installed, never
  recomputed: a CPU recompression of a CUDA payload is not byte-identical.
- `tests/test_d0s_v2.py` (CPU) covers:
  - a tiny trained fixture of both forms derives a /2 that the reader, the merged-arena writer and refusal 13 accept;
  - two runs give byte-identical manifests;
  - each broken link refuses.
