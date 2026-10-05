### K30 registered: the int4-b32 split-K R term on one rented NVIDIA L4 (bench and tests only)

- **Why.** grouped-nf4-gemm's R term is on by default for every part with 64 SMs or fewer, and its only receipt is an RTX
  A2000 sweep. The A2000 is a correctness-only testbed, so the term has no admissible speed evidence (the A2000-timing
  audit, e4b#1133 / grouped-nf4-gemm#475). The owner asked for an L4 read.
- **The lane** (prereg and rule in grouped-nf4-gemm `kernel/PREREG-k30-splitk-r-term-l4.md`) runs the unchanged 48-cell
  `sk_sweep.py` twice on one L4 (58 SMs), after three correctness gates. It reads KEEP if the R-aware pick's summed time
  is <= 0.97 of the N-only pick's with no cell > 1.02 worse, and OFF otherwise.
- **Here:** `bench/k30/k30_drive.sh` (controller, K20's pattern), `bench/k30/k30_run.sh` (box; `K30_PROVE=1` for the
  proving rental, `K30_REHEARSAL=1` for the $0 A2000 rehearsal), `bench/k30/staged.sha256`, and
  `tests/test_k30_staged_pin.py` (pins, and the gates ordered before any timing).
