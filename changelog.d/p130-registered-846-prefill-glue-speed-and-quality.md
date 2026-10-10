### P130 registered (#846): do #1583's prefill knobs make the served 512-token prefill faster, and does P1's arithmetic cost quality? (bench and tests only)

- **Why.** About a quarter of Qwen3-30B-A3B's 512-token prefill forward on SC2e's int4 stack is eager glue
  (`bench/decode-census/PREFILL.md`), and every admitted prompt's forward stalls the decoders behind it. #1583 added two
  opt-in knobs: `E4B_FUSE_PREFILL_GLUE` (P1, the decode folds above 64 rows; it changes bf16 arithmetic) and
  `E4B_PREFILL_LEAN_DISPATCH` (P2, K19's lean dispatch on prefill rows; bit-identical). Nothing licenses either as a
  default yet. Not TC1's predictions P130–P133, which used the same labels.
- **The box** (`bench/p130/p130_box.py`): SC2e's int4 stack in eight processes on one RTX 5090, P1 set per process in
  the order 0 1 1 0 0 1 1 0. **Phase A** captures the served first-chunk prefill graph (T = 512) with P2 off and on in
  every process, checks the two bitwise (FUNCTION), digests the default's output across processes (DETERMINISM), and
  times them interleaved. **Phase B** is P117's teacher-forced instrument with one subject, P1, against R's own floor
  (half, chunk, rev) and the scale mutant.
- **The rule** (`bench/p130/p130_reduce.py`, 40 self-test cases):
  - P2 is licensed iff the 95 % t-interval of its per-process ratio stays under ×1.01;
  - P1 is licensed iff the interval of its per-pair gain clears ×1.03 AND Phase B is AT_PARITY on P117's bar;
  - a capture refused under a knob, a P2 FUNCTION failure, cross-process nondeterminism or noise holds the knob.
  Each flip is its own later PR. P1's default is scoped to `qwen3_moe`, the family read.
- **The pre-rental fetch gate** (`bench/p130/p130_fetch_gate.py`, DQ11's launch-gate pattern): before the rental
  controller quotes, everything the box fetches is resolved from a CPU host with the box's own clients. That covers:
  - the checkpoint's files, with sizes and hashes;
  - the corpus;
  - both commits by SHA;
  - every PyPI wheel, picked by pip for the box's platform.

  A missing, unauthorized or mismatched entry refuses. `p130_drive.sh` stages nothing without a passing, fresh report
  for the launch commit.
- **Proof** on Qwen3-30B-A3B at the proof's sizes (four processes; guard 1.25 h); reading guard 2.0 h; lane ceiling
  $5.50. Nothing rents before this merges, the maintainer ACKs and SD2's `sd2-5090-1` read has finished.
