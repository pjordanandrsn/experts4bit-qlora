### Read: TC1 amendment 54 -- neither remedy moves torch 2.8's step on a host that is not host-bound (P137, P138 FALSIFIED; P139, P140 HELD)

- `tc1-5090-111` ($1.41, a Threadripper PRO 7965WX). At e4b's defaults in torch 2.8 the matched arm stepped 10.50 s with device time 99 % of
  the step, against 12.55 s on amendment 53's host. Neither remedy had host time to recover:
  - grouped-nf4-gemm's bucket ladder: 1.010;
  - a larger cuBLASLt heuristics cache: 0.996.
- Held-out within 0.0002; the ladder's peak +0.34 GB. Row `e4b.train.pad-ladder.torch28.qwen3.5090.2026-10-06`.
- Reported, not scored: the ladder cut `aten::bmm`'s CPU self time per call about tenfold, so the per-shape cost is real. It matters where
  the host is the bottleneck. The next registration reads the registered candidate, fewer and wider buckets, on a host where the defaults
  are host-bound. STATUS says torch 2.8's host-side share depends on the host.
