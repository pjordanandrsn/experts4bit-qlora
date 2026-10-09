### `E4B_INT4_TILE_PROGRAMS=P`: the one-launch cumsum tile table above 256 routed rows, split over P programs (opt-in; #846)

- **What changes, opt-in.** A device-grouped call with 257–1,024 routed rows (a decode step above 32 rows at top-k 8)
  can build its tile table with grouped-nf4-gemm's cumsum rank. When it does, `E4B_INT4_TILE_PROGRAMS=P` (an integer
  from 2 to 64) passes `programs=P` to `build_group_tiles_fused` (grouped-nf4-gemm #524). Each of P programs then ranks
  a slice of the experts instead of one program ranking them all. The tables are the same integers at every P, so
  outputs are bit-identical.
- **Default.** `1` (also unset or empty) calls the builder exactly as before, with no `programs=` keyword. Calls that
  take no cumsum table are unchanged: at most 256 rows, `E4B_INT4_WIDE_TILES=0`, and prefill chunks.
- **Refusals.** `P > 1` is refused where the cumsum table is built if the installed grouped-nf4-gemm's builder has no
  `programs=`, which is detected from its signature, never from a version. Anything that is not an integer from 1 to
  64 is refused.
- **Why.** Lane P122 read the one-program chunked table at 2.51 ms of Qwen3-30B-A3B's 64-row decode step. A P-lane will
  read the split before any default moves. It takes effect with the grouped-nf4-gemm release carrying #524, after
  0.44.0.
