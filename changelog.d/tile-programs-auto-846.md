### `E4B_INT4_TILE_PROGRAMS` defaults to `auto`: the one-launch tile table over 4 programs, inside the size P126 read (#846)

- **What `auto` does.** It is the default, also when the variable is unset or empty. It splits the cumsum tile table over
  **4 programs** when the installed grouped-nf4-gemm's `build_group_tiles_fused` takes `programs=`, detected by
  capability, and the table is within `next_pow2(E) × next_pow2(R) ≤ 128 × 512`. Otherwise it keeps the one-program
  table.
- **The evidence.** Lane P126 read DEFAULT_ON_4 on Qwen3-30B-A3B int4 on an RTX 5090: the captured 64-row decode step was
  about 15.5 % faster with every token identical (`e4b.serve.p126.tile-programs.qwen3-int4.5090.2026-10-09`). P = 8 was
  not licensed.
- **When it takes effect.** With a grouped-nf4-gemm release carrying #524. On 0.44.0 and older, `auto` changes nothing.
- **The way back.** `E4B_INT4_TILE_PROGRAMS=1` restores the one-program table, called exactly as before. An integer
  from 2 to 64 still forces that many programs at any size.
