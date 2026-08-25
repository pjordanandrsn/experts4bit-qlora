# RESULTS — F1 Stage B: PARTIAL, and it ships (74.2 → 94.2 tok/s)

Measured 2026-08-25 under PREREG-f1-stageB. Receipts in
`receipts-f1/stageB/` (RTX 5090, driver 580.119, torch 2.13.0+cu130 /
triton 3.7.1; e4b `e6554fd`, gnf4 `1517955`; every arm run from a
verified-clean merged-main tree; instance destroyed, vast and DO both
verified zero).

```
F1 STAGE B VERDICT: PARTIAL
  B1 (compile-layers default) step 10.62 ms in (10.5, 12.0]
  (94.2 tok/s, 2.87 ms removed, A/A 0.00)
```

| arm | step | tok/s |
|---|---|---|
| B0 (anchor) | 13.485 ms | 74.2 |
| B0 (repeat) | 13.485 ms | 74.2 |
| **B1 (compiled body)** | **10.620 ms** | **94.2** |

**PARTIAL ships.** The prereg's condition is that the arm's own A/A
spread be under half the measured gain: spread **0.00 ms** against a
**2.87 ms** gain. The PASS bar (≤ 10.5 ms) was missed by 0.12 ms —
1.1% — and the honest reading is that the bar was set from an estimate
of what fusing RMSNorm and RoPE would return, and the estimate was
essentially right: Stage A predicted 1.85 ms from those two mechanisms,
and the arm returned 2.87 ms, i.e. inductor also fused work the census
attributed elsewhere.

## Every registered refusal, discharged

| refusal | result |
|---|---|
| token identity vs B0 | **identical** — 127 tokens, 78 distinct values, equal length, element-wise equal |
| A/A before A/B | **0.00 ms** spread across two B0 arms |
| capture neutrality | graph arm, 127 steps both arms |
| no recompile in window | **0** dynamo recompiles inside the timed window (B0 and B1) |

The identity check is recorded with its cardinality on purpose: a
127-entry, 78-distinct trace cannot pass identity vacuously, which an
empty or degenerate trace would.

## The bug that had to be fixed first

B1 could not compile at all on the first attempt. Inductor died
emitting **our own** fp8 paged-decode triton kernel through its
user-kernel path (`Loop-carried variable m_i has initial type fp32 but
is re-assigned to fp64`) — a kernel the arm's registered scope
excludes. `_b1d_stage_a` unwraps the timed attention shim back to
`._orig` to keep `cudaErrorStreamCaptureInvalidated` off the first
attention call, and `._orig` is the RAW shim, so the unwrap discarded
the `dynamo.disable` that `--compile-layers` had applied. Restoring it
(#235) fixed the arm without re-scoping it.

## Ladder

14.1 → 20.1 (collapse) → 65.8 (graph loop) → 74.3 (kernel configs) →
**94.2 (compiled layer body)**. Roofline ~480; the goal is 425.

## Next, per the registered order

B2 was registered before any B1 number was seen, conditioned on B1
landing short of PASS. It did. B2 targets the remaining **1.58 ms** of
fp8 KV work (`_write_side`'s two symmetric copies, 1.08 ms;
`quantize_kv_fp8`'s abs/amax/where/div/gt, 0.50 ms) — our own code,
inside the dynamo-disabled shim, which compile cannot reach by
construction. At the new 10.62 ms step, removing it would land near
9.0 ms ≈ 111 tok/s.
