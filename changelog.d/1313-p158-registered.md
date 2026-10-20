### P158 registered (#1313): K39's served read, the K25 decode `lut="prmt"` as Qwen3-30B-A3B's batched-decode expert GEMM

P156 read K39's decode GO in a microbench (1.1755x on the two K25 GEMMs, lower bound 1.1728; at least 1.1103 under its
sampled clock range). P158 reads it in the served step, by P153's served form.
- **Arms:** S0, `E4B_K25_LUT=tree` (named), and S1, `prmt` (e4b#1862), on the shipped default server with the int8 head
  at its qwen3_moe default; a warm-up process, then ABBA twice through W16 and W1 at five reps.
- **The rule is fixed before data:** FLIP needs g16, the min of four W16 pairs, >= 1.03 and g1 >= 0.99 (W1 is a guard:
  K25 is not on T == 1), on unrounded minima.
- **Identity, null first:** the arms are bit-identical by contract, so the gate is token identity, not a quality band.
  S0 against S0 across processes is checked first and VOIDs the read ("tree not token-deterministic") before any S1
  difference can be a contract finding against K39.
- **Engagement:** after the build, K25's call record reads exactly 960 under the arm's decode and the bandwidth GEMV's
  tally exactly 480 (48 layers x five forwards per row count: the residual licence's two probes, two warm-ups, one
  capture), derived from the code, tested, and confirmed on a $0 layer-cut build before any rental.
- **Clocks:** a 500 ms sampler in its own process during every timed block (100 ms cost +0.25 % on a $0 A2000 smoke;
  500 ms did not) labels a pair "clock-confounded" beyond 2 % either way, beside the verdict, never ruling.
- **Box:** one RTX 5090 at >= 575 W (power limit and enforced limit), grouped-nf4-gemm v0.50.0 (41ef086e), premise 49
  tests (K39's compiled contract and e4b's served-route contract among them), guard 1.5 h with the pre-install fit at
  25 % headroom from two measured served reads, worst $1.69.

A FLIP opens a separate PR to make `prmt` qwen3_moe's K25 decode by default. Files: `bench/p158/` and
`tests/test_p158.py`.
