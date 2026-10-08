### `E4B_INT4_WIDE_TILES` defaults to `auto`: the one-launch tile table above 256 routed rows, where lane P122 read it faster (#846)

- **What changes.** `auto`, now the default and also used when the variable is unset, builds the device tile table of
  a device-grouped call with 257–1,024 routed rows in one launch, using grouped-nf4-gemm's cumsum rank. It does so only
  when both of these hold:
  - the installed `build_group_tiles_fused` takes `rank=` and `rchunk=` (grouped-nf4-gemm #515 and #519). This is
    detected by capability, never by version string.
  - the table is no larger than the one read: `next_pow2(E) × next_pow2(R) ≤ 128 × 512`.

  Otherwise the chained builder runs, as before.
- **Why.** Lane P122 (`e4b.serve.p122.wide-tiles-chunked.qwen3-int4.5090.2026-10-08`) read Qwen3-30B-A3B's 64-row
  decode step 4.3 % faster with #519's chunked table, with identical tokens (DEFAULT_ON). Without the chunks the table
  was 1.44× slower (lane P120), so `auto` keeps the chained builder on a kernel package that has `rank=` but no
  `rchunk=`.
- **When it takes effect.** grouped-nf4-gemm #519 is on grouped-nf4-gemm's main but in no release yet. Until a
  grouped-nf4-gemm release carries it, `auto` finds no `rchunk=` and nothing changes.
- **Other values.** `1` still forces the one-launch table at any size, and is refused without `rank=`. `0` keeps the
  chained builder, as before P122. Outputs are bit-identical every way: the tables are the chained builder's integers.
