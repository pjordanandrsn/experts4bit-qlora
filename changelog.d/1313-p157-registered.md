### P157 registered (#1313): what bounds K25, its stages knocked out one at a time at today's plan on Qwen3-30B-A3B's W16 shapes

A microbench premise for stream (B). P155 found no plan lever and P156 a decode lever; P157 reads how K25's DRAM stream
and its on-chip work combine, and prices each stage's removal, before the next decode lane is chosen.
- **Arms** (`bench/p157/p157_arms.py`, K25 copied verbatim plus one knock-out each): A0 (K25 as served), T2 (the
  self-pair), A0p (the placebo, SASS identical), A1 (no select tree), A2b (no MMA or staging), A12 (both), A3 (loads
  only), AL (A3 at ~2 % of the bytes), a residency ladder AC/AC4/AC16/ACX (the whole K loop at ~2/7/28/100 % of A0's
  DRAM bytes), ACX0 (the ladder unrotated, bitwise to K25) and AE (empty). A2a, a PTX knock-out of the MMA, is not
  built: a faithful fold needs 24 `lop3` against 8 HMMA an iteration.
- **The read:** ceilings per stage; A12's additivity; r_dram = (full - max(mem, onchip)) / min(mem, onchip) from the
  arms, each difference taken inside one cubin, banded THROUGHPUT_BOUND / MIXED / LATENCY_BOUND / NOT_ADDITIVE /
  UNRESOLVED; the ladder checks the band out of sample, its doses measured from AC's. The SE is the median's.
- **The instrument:** 70 paired reps (5 whole Williams squares of the 14 arms) on S0's real router picks, a spacer graph
  before every timed replay, occupancy pinned to A0's by dynamic shared memory (driver-checked, read back from the graph
  nodes), SASS gates from the cubins that ran (vendored `sass_count.py`), NVML clocks and power per arm, Nsight DRAM
  bytes per ladder arm where the host allows counters.
- **Box:** one RTX 5090, grouped-nf4-gemm v0.50.0 (its kernel files byte-identical to P156's pin), guard 1.5 h
  ((boot + fit) x 1.25), worst $1.69.
  The $0 A2000 shakedown's record is a sha-pinned test fixture.

Files: `bench/p157/` and `tests/test_p157.py`; P157's driver joins the lane-liveness tests.
