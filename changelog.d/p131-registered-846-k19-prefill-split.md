### P131 registered (#846): K19 over a split tile table on the served first-chunk prefill graph (bench and tests only)

- **Why.** A $0 shape census of Qwen3-30B-A3B on SC2's prompts found a 512-token chunk's rows bimodal: one expert takes a
  median 398 of 4,096 routed rows. On K19's 16-row table that expert runs as ~25 tiles. `E4B_K19_PREFILL_BIG_M` (#1625,
  over grouped-nf4-gemm#542) runs each expert's full 32- or 64-row tiles at a tall M tile and keeps the 16-row padding.
- **The box** (`bench/p131/p131_box.py`, P130's by named substitution).
  - Eight identical processes at the release's prefill defaults.
  - Each process captures the first-chunk prefill graph at the knob's 0, 32 and 64 and times the three interleaved.
  - It checks FUNCTION against 0 on every window, and that a 64-token forward (a 64-row decode step's 512 routed rows)
    shows no split and the same K19 calls as m0's, whichever table `E4B_INT4_WIDE_TILES` picks (on Qwen3-30B, the wide
    one-launch table).
  - Process 1 also runs P117's teacher-forced instrument with the subjects S32 and S64.
- **The rule** (`bench/p131/p131_reduce.py`, 37 self-test cases).
  - An arm is LICENSED iff it is at least x1.02 faster (95 % interval) and its FUNCTION is bitwise, or, failing that, its
    Phase B subject is at parity.
  - When both are licensed, the paired m32-vs-m64 interval picks the default, with m32 on a tie.
  - The predictions come from a weight-pass cost model over f in [0.3, 0.8], with the builder's measured 48-vs-39 ops.
- **Harness.** P130's run, drive, fetch gate and tests by named substitution; `staged.sha256` pins p131's files and
  P130's box.
- **Real records.** `tests/fixtures/smoke1625/pass2_summary.json` is #1625's A2000 serve-smoke receipt, byte for byte
  (sha256 checked). `tests/test_p131_real_records.py` runs the decode check on its K19 records first, then synthetic
  departures from them.
