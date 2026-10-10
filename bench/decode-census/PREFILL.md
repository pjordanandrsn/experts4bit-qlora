# Lane 2 design note (a): where e4b's 512-token prefill forward spends its time

**Zero rental; for review before any registration** (#846, lane 2). This follows `README.md` in this directory: at
C = 64, SC5's TPOT carries about 10 ms per token of prefill stall, and each admitted 512-token prompt's forward is the
stall. `prefill_census.py` computes every number below from committed receipts, and `tests/test_prefill_census.py` keeps
this file equal to its output.

**Sources.**
- P119's eager profile of one 512-token prefill: `prefill.p512_on`, with logits for the last token only, as served.
  The stack was e4b `caac7f08` and grouped-nf4-gemm `b4f93f1c` on an RTX 5090. Kernels are classed by P119's own
  frozen table.
- SC2e's served prefill steps (`verdict_sc2e.json`, arm `s16`).
- SC5's C = 1 TTFT on 0.52.0.

## 1. The forward, by class

| class | device ms | share |
|---|---:|---:|
| int4 experts (K19 grouped GEMM) | 19.95 | 49.5% |
| elementwise (PyTorch eager kernels) | 10.59 | 26.3% |
| dense GEMM (bf16 attention projections, router, head) | 5.66 | 14.1% |
| routing | 2.10 | 5.2% |
| attention (flash) | 1.50 | 3.7% |
| other (`_swiglu_rows`, misc) | 0.30 | 0.7% |
| device-to-device copies | 0.16 | 0.4% |
| copies, sampling (under 0.05 ms each) | 0.01 | 0.0% |
| **total (eager, last-token logits)** | **40.28** | |

The elementwise class is PyTorch eager kernels around the fused ones; by kernel name:

| elementwise, by kernel name | device ms | launches |
|---|---:|---:|
| copies, casts and cat | 4.48 | 1510 |
| index, gather and scatter | 2.24 | 491 |
| binary arithmetic (adds, muls) | 1.53 | 1077 |
| norm math (eager RMSNorm: mean, pow, rsqrt) | 1.31 | 579 |
| softmax, where, fill | 0.43 | 440 |
| other elementwise | 0.61 | 350 |

Routing:

| routing, by kernel | device ms |
|---|---:|
| radix sort (`radixSortKVInPlace`) | 1.20 |
| other routing (top-k, scans, combine) | 0.90 |

## 2. Closing it against the served prefill

- **The served forward.** SC2e's served forward, device p50, is 39.97 / 40.02
  ms in draws 1 / 2. The residual against this eager total is -0.30 /
  -0.26 ms, from a different box and graph replay against eager.
- **The served prefill step** is 48.79 / 48.74 ms. That is the forward plus
  8.82 / 8.72 ms of
  riding decode (9 / 8.5 rows p50) and host. The direct
  stall it puts on the decoders is 40.49 / 40.54 ms.
- **A consistency check, not a split.** SC5's C = 1 TTFT p50 on 0.52.0 is 43.52 ms: one forward of about
  40 ms plus the first decode and the host.
- **The full-vocabulary logits** P119 also profiled (`p512_off`, 41.21 ms) cost
  0.94 ms more. Serving keeps the last token only.

## 3. Rates on the same profile

- **The bf16 attention projections** (the two 128×128 CUTLASS kernels, 96 launches: q|k|v and o in 48 layers):
  0.928 TFLOP in
  4.83 ms, which is **192 TFLOPS**. That is this card's achieved dense bf16 rate, read from the
  same profile.
- **The int4 experts (K19):** 1.855 TFLOP in 19.95 ms, which is
  **93 TFLOPS**, 48% of that dense rate. At the dense rate they would take 9.66 ms.
- **Expert weight bytes:** P100's 87 distinct experts per layer at 4.50 bpw is
  11.1 GB, which is 6.19 ms at 1792 GB/s.
- So the prefill experts are compute-bound, and run at under half the card's achieved dense rate.

## 4. What this suggests for lane 2 (for review; nothing is registered)

At C = 64, each millisecond off this forward is about 0.25 ms of TPOT: (C − 1) / 256 per admitted prompt.
1. **The eager glue:** 10.59 ms (26%) across about
   4447 launches, plus the 1.20 ms
   radix sort in routing.
   - Copies, casts and `cat` are the largest part.
   - The decode path's fused RMSNorm (`experts4bit_qlora/engines/glue_fuse.py`) hands any call over 64 rows back to
     the eager chain by design ("Prefill ... keep the original chain"). The prefill therefore runs PyTorch's norm math.
   - **The first candidate:** it is a code change with a bitwise or teacher-forced check, not a kernel project.
2. **K19 at the prefill shape** (about 32 rows per expert): 19.95 ms, at 48%
   of the dense rate. The ceiling on what a prefill-shaped tile configuration could recover is
   10.29 ms.
3. **The bf16 projections** already run at the card's dense rate. int4 would save bytes, and the compute-bound
   prefill does not need them.
