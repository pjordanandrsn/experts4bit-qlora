### `E4B_ATTN_INT4_WIDE` defaults to `auto`: the attention projections of a 17–64-row decode step on the int4 small-M GEMM, where lane P124 read them faster (#846)

- **What changes.** `auto`, now the default and also used when the variable is unset or empty, serves the attention
  projections of a 17–64-row decode step from the int4 grid, using the K16 small-M GEMM at a 32- or 64-row tile,
  instead of a cached bf16 copy on cuBLAS. It does so only when both of these hold:
  - the K16 route is on (`E4B_ATTN_INT4_SMALLM`, `auto` by default);
  - the installed `gemm_int4_b32_smallm` takes `block_m=` (grouped-nf4-gemm #522), detected from its signature,
    never from a version string.

  Otherwise those rows take the cached bf16 matmul, as before. When K16 is on but `block_m=` is missing, one line
  says so.
- **Why.** Lane P124 (`e4b.serve.p124.attn-int4-wide.qwen3-int4.5090.2026-10-09`) read DEFAULT_ON on Qwen3-30B-A3B
  int4. With SC2e's stack on one RTX 5090, the captured 64- and 32-row decode steps were 3.2–3.8 % faster, and the
  teacher-forced NLL stayed inside P110's bar (+0.0011 nats at 64 rows, +0.0014 at 32). Other models ride that read.
- **When it takes effect.** With the grouped-nf4-gemm release that carries #522 (0.44.0, pending). v0.43.0 does not
  carry it: there `auto` finds no `block_m=` and nothing changes.
- **Other values.** `1` still requires the route, and is refused without the K16 route or without `block_m=`. `0`
  keeps the cached bf16 matmul, as before P124.
- **Memory.** The route's split-K workspace is shared by every projection of one width. P124 measured 11.5 MiB on
  Qwen3-30B-A3B, against 1.81 GB a step for the bf16 copy the route stops reading. About 4.5 MiB of that belongs to the
  q/k/v widths built before the fusion pass replaces them. The serve estimate now prices the workspace once per
  projection width it sees.
- **Tests.** P124's premise ran `tests/test_int4_attn_wide.py` at its registered bytes. Those bytes now live in
  `bench/p124/test_int4_attn_wide.py`, which P124's driver stages and its staged-pin test checks, so the live tests can
  follow the new default.
