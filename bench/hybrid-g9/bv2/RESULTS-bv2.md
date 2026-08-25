# RESULTS — BV2: the curve tops at 123.7 tok/s; system-425 not claimed

Measured 2026-08-25 under PREREG-bv2-curve. Receipts in
`receipts-bv2/` (box 13, anchor-compliant at 7.34 ms; box 12's
attempt is preserved as the anchor-gate REFUSAL it was; both
instances destroyed, vast zero).

```
BV2: CURVE
  system throughput tops at 123.7 tok/s (B=16) -- below the 425
  line; the campaign reports the measured maximum and stops claiming
```

| B | step (eager scheduler) | aggregate tok/s |
|---|---|---|
| 1 | 59.07 ms | 16.9 |
| 2 | 75.47 | 26.5 |
| 4 | 83.07 | 48.2 |
| 8 | 107.77 | 74.2 |
| 16 | **129.37** | **123.7** |

Every refusal clean: per-B A/A token-identical, cross-B row-0
identity held at every batch size, anchor 7.34 within 0.7% of the
certified class, traces non-degenerate.

## The structural finding

The production scheduler path is **host-bound**: eager B=1 runs
59.1 ms against the 7.34 ms graph loop on the SAME box, and across
boxes the eager curve swings ~30% with host single-thread speed
(box 12's refused attempt: 98.3 ms at B=16 → ~163 aggregate, on a
slower-GPU/faster-CPU host) while the graph path holds within 1%.
Batch aggregate throughput on this stack is a property of the host
CPU, not the GPU. **A B>1 CUDA-graph decode loop is the single
registered-able lever for system throughput** — and the
device-grouping machinery (S3, bitwise-certified, capture-safe) plus
the graph-mode KV appends are exactly its prerequisites, both already
merged.

## Fresh census (knob in certified state, eager-profiled, disclosed)

Device 10.81 ms/step at 100.2% coverage: NF4 GEMV 3.07, attn-proj
cuBLAS 1.92, elementwise 4.80-eager, fp8 other 0.55, router 0.31,
memcpy 0.15. On the graph basis the step is 7.35; the census's eager
inflation is disclosed and its SHARES guide F2: attn-proj is now the
largest non-NF4 item.

## Campaign statement

- Single-stream: certified default ~136 tok/s; 151.7 with the
  certified-available dot-pad knob; 425 REFUTED (RESULTS-s3).
- System: **123.7 tok/s measured maximum on the production path**;
  425 not claimed. The registered route to a real system number is
  the B>1 graph, for which every prerequisite now exists.
