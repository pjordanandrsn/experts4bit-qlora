# PREREG — F1: the elementwise/copy block at B=1

Registered 2026-08-24, before any F1 measurement. Basis: the K5 cycle
closed the kernel lane (gnf4#251, STRUCTURE-REFUTED) and its graph
anchor established that decode is **device-bound under graph replay**
— the b1d profile's ground-truth device total is **12.55 ms/step**
against a **13.46 ms/step** wall at the certified 74.3 tok/s, i.e.
~93% of the step is kernels on the device.

## Stage A — the census (complete; attribution remains)

`step_budget.py` (this directory, self-tested 9 ways) parses a
step_decomp `--host-brackets` kernels table into a per-step device
budget and gates coverage against the table's own `Self CUDA time
total` footer.

**Correction, made before any F1 treatment was measured.** The first
draft of this prereg reported a 79.7%-coverage REFUSE and blamed a
row-limited table. That diagnosis was wrong: the parser deduped rows
by the profiler's *clipped* name, and this table prints 13 distinct
`vectorized_elementwise_kernel<4, at...` instantiations under one
label — 22 of 39 kernel rows were silently dropped (Bugbot, e4b#230).
With clipped labels aggregated the table covers **99.9%** and needs no
re-profiling. The census below is therefore the real one, and the bars
in Stage B are set against it rather than against the undercount.

Census (b1d eager profile, 12 active steps, coverage 99.9% of a
12.55 ms/step device truth; step wall at the certified 74.3 tok/s is
13.46 ms, so ~93% of the step is device work):

| kind | ms/step | launches/step | share |
|---|---|---|---|
| matmul (nf4 GEMV 4.84 + cublas gemv 1.93) | 6.77 | 337 | 54.0% |
| **elementwise** | **4.73** | **3901** | **37.7%** |
| other (fp8 combine/paged-decode) | 0.57 | 96 | 4.5% |
| router (topk/sort) | 0.32 | 96 | 2.5% |
| memcpy DtoD | 0.15 | 192 | 1.2% |

The block averages **1.21 µs per launch** across 3901 launches — at or
near this GPU's minimum kernel duration, so the block is
launch-quantum-bound, not arithmetic-bound: fusing K kernels into one
recovers nearly (K−1) quanta regardless of the math inside them. That
is the mechanism F1 tests.

Stage A closes when every elementwise kernel above 50 µs/step is
attributed to a call site (profiler `with_stack`), because
the treatment differs by cause and the two are not interchangeable:
- **dtype churn** (e.g. `hot_residency.py:92` `gate.float(), up.float()`
  then `.to(gu.dtype)` at :95) is fixed by dtype discipline, no kernel;
- **genuine per-element math** (rmsnorm pow/rsqrt/mul/div, silu×gate,
  residual add) is fixed by fusion.

## Stage B — treatment, registered bars

Bars are set on the **graph-replay basis** (AMENDMENT-k5-graph-timing;
the eager harness measures host, not kernel) at B=1 on one box, with
the b1d greedy token-identity gate unchanged (fusion must not move a
single token; any divergence is a REFUSE, not a disclosure).

- **PASS**: end-to-end step ≤ 10.5 ms (≥ 95.2 tok/s), i.e. ≥ 2.96 ms
  removed — 63% of the 4.73 ms block.
- **PARTIAL**: 10.5–12.0 ms; the treatment ships only if its own A/A
  spread is < half the measured gain.
- **REFUTED**: > 12.0 ms (< 11% gain) — the block is not addressable
  by this mechanism and F1 closes.

## What this lane CANNOT do (registered so the terminus is honest)

The 425 tok/s goal needs **2.35 ms/step**. Charging every non-matmul
cost to zero (elementwise 4.73 + other 0.57 + router 0.32 + memcpy
0.15) *and* the nf4 GEMV down to its K4 loads floor (4.84 → 0.96)
leaves **3.81 ms/step ≈ 262 tok/s** — the optimistic bound of the
entire kernel+fusion program, with the attention projections' 1.93 ms
of cublas GEMV left untouched because they are already near their own
bandwidth floor at B=1. (The complete census moved this bound from the
draft's ~290; the conclusion is unchanged.) **425 single-stream is therefore
not reachable by making each step cheaper; it requires emitting more
than one token per step.** Speculative decoding is the only registered
route across, and it gets its own prereg whose FIRST deliverable is an
acceptance-rate measurement (a draft that is not accepted ~2× buys
nothing). F1 is registered as a rung on that path, not as a route to
425 by itself.

## Verdict calculator

`f1_verdict.py` (self-tested both directions) reads the Stage-A census
and the Stage-B arms; receipts in `receipts-f1/`.
