### `E4B_FUSE_PREFILL_GLUE` is on by default for `qwen3_moe`: the decode folds take prefill rows (#846, P130)

- On `qwen3_moe`, a prefill's norms, residual-and-norm and q/k norm with the rotary now run the fused kernels decode
  already uses, instead of transformers' eager chain. This changes prefill arithmetic: decode's single bf16 rounding
  replaces transformers' two.
- **The read** (`bench/p130/RESULTS-p130.md`, one RTX 5090, Qwen3-30B-A3B int4): ×1.118 [1.116, 1.119] per 512-token
  first-chunk prefill replay (4.02 ms of 38.2), and at parity on P117's teacher-forced instrument (bias −0.0035 nats,
  inside the floor's 0.0125 bar).
- **`serve_paged`'s default is per family:** unset means `1` on `qwen3_moe` and `0` on every other family. `1` or `0` is
  explicit, and `0` restores the eager chain. `/health`'s `engine.prefill_glue` reports the mode and its source. The
  library's fold functions keep unset = `0`, so bench harnesses calling them directly do not move.
- Register rows `e4b.serve.p130.prefill-glue-speed.qwen3.5090.2026-10-10` and
  `e4b.serve.p130.prefill-glue-quality.qwen3.5090.2026-10-10`.
