### Fixed

- P224 kit, per Amendment 3: the arena is baked on the box with grouped-nf4-gemm's `bake_nf4` from the HF snapshot
  and its provenance (source commit, nf4/64 quantizer) is checked before any server process; canary B (REF-DQ, PA-DQ)
  is dropped and canary A must record the reference kernel and an fp32 expert absmax; the reduce gets `pins.json`;
  the fixture trains take `--fix expert_kernel=grouped_nf4` and run whatever the reduce outcome. Each kit script gains
  `--check-args`, and a test runs every runner call through it; another runs the paged refusals on an index a real
  `bake_nf4` writes on CPU.
