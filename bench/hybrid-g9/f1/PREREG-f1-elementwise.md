# PREREG — F1: the elementwise/copy block at B=1

Registered 2026-08-24, before any F1 measurement. Basis: the K5 cycle
closed the kernel lane (gnf4#251, STRUCTURE-REFUTED) and its graph
anchor established that decode is **device-bound under graph replay**
— the b1d profile's ground-truth device total is **12.55 ms/step**
against a **13.46 ms/step** wall at the certified 74.3 tok/s, i.e.
~93% of the step is kernels on the device.

## Stage A — a census that is allowed to refuse

`step_budget.py` (this directory, self-tested 7 ways) parses a
step_decomp `--host-brackets` kernels table into a per-step device
budget. Run against the existing b1d receipt it **REFUSES**: the
kernel view covers 79.7% of the profiler's own `Self CUDA time total`
because that table is row-limited. So the first F1 deliverable is a
**full-coverage profile** (row_limit raised until coverage ≥ 90%), not
a treatment.

Provisional, disclosed-as-lower-bound shares from the 79.7% view:

| kind | ms/step | launches/step |
|---|---|---|
| matmul (nf4 GEMV 4.84 + cublas gemv 1.56) | 6.40 | 193 |
| **elementwise** | **2.22** | **1591** |
| other (fp8 combine/decode) | 0.91 | 289 |
| router (topk/sort) | 0.32 | 96 |
| memcpy DtoD | 0.15 | 192 |
| *unattributed (row-limited)* | *2.55* | — |

Stage A closes when coverage ≥ 90% AND every elementwise kernel above
50 µs/step is attributed to a call site (profiler `with_stack`), because
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

- **PASS**: end-to-end step ≤ 12.0 ms (≥ 83.3 tok/s), i.e. ≥ 1.46 ms
  removed — 66% of the visible elementwise block.
- **PARTIAL**: 12.0–12.9 ms; the treatment ships only if its own
  A/A spread is < half the measured gain.
- **REFUTED**: > 12.9 ms (< 3.9% gain) — the block is not addressable
  by this mechanism and F1 closes.

## What this lane CANNOT do (registered so the terminus is honest)

The 425 tok/s goal needs **2.35 ms/step**. Charging every visible
non-matmul cost to zero *and* the nf4 GEMV down to its K4 loads floor
(4.84 → 0.96 ms) leaves ~3.4 ms/step ≈ **290 tok/s** — and that is the
optimistic bound of the entire kernel+fusion program, with the
unattributed 2.55 ms assumed free. **425 single-stream is therefore
not reachable by making each step cheaper; it requires emitting more
than one token per step.** Speculative decoding is the only registered
route across, and it gets its own prereg whose FIRST deliverable is an
acceptance-rate measurement (a draft that is not accepted ~2× buys
nothing). F1 is registered as a rung on that path, not as a route to
425 by itself.

## Verdict calculator

`f1_verdict.py` (self-tested both directions) reads the Stage-A census
and the Stage-B arms; receipts in `receipts-f1/`.
