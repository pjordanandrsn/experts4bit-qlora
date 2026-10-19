# `p131-prove-1`: the proof (PROVED)

One RTX 5090 (cc 12.0, driver 610.57.04) on an AMD EPYC 7D12 host (64 CPUs, 251 GB RAM): Vast instance 56667234 on
machine 150168 at $0.7504/h, 2026-10-19 04:04Z to 04:48Z. Cost $0.661 (guard 1.25 h). Code: e4b `c7a40682` (the
registered launch commit under Amendment 1), grouped-nf4-gemm `57f53e19`. The proof's sizes: four processes, 4 speed
windows, 3 timed rounds after 1 warm-up, Phase B over 16 windows of 512 prompt tokens and 32 positions.

**PROVED.** The self-test read 37/37 and the premise 9 passed. FETCH and BAKE were OK, and four processes exited 0.
- Refusals: none. DETERMINISM: no differences. The decode check: untouched.
- Both arms read BITWISE.
- Phase B: rep, rev, S32 and S64 read exactly 0, and the mutant was caught (bias 1.08). The floors were B 0.0126 /
  S 0.0374.

The proof's verdict, **M32_NOISY M64_NOISY DEFAULT_NONE**, is not a reading. It came from the m0 span guard (x1.0475 >
x1.02): process 1 ran about 4.7 % faster on every arm than processes 2-4, which agreed to 0.1 %. The receipts carried
no GPU clock, power or temperature, so the effect could not be attributed. Amendment 2 (#1800) answered it before any
reading data: a warm-up process 0, the GPU state per process, and a reading guard of 2.5 h sized from this proof's
measured loads (430-438 s a process).

The maintainer re-derived the proof from the private receipt store with the reducer at `c7a40682`. `SHA256SUMS`
covers every file here.
