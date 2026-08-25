# RESULTS — S3: grouped verification measured directly. REFUTED, 0.66×.

Measured 2026-08-25 under PREREG-s3-grouped-verify. Receipts in
`receipts-s3/stageA/` (box 10: RTX 5090, e4b `28a9fac` + gnf4
`73652c2` mains; instance destroyed, vast zero).

```
S3 VERDICT: REFUTED
  best K=64: T_pred 0.66x < 1.2x: the graphed verify is too
  expensive at every measured K
```

## Every gate passed before a single timing was read

| gate | result |
|---|---|
| grouping parity (device vs host builder) | exact — 23 CPU tests, CI-gated in gnf4 |
| numeric parity, device-grouped vs eager-grouped | **max\|Δ\| = 0.000e+00, tokens identical** — bitwise, stronger than required |
| sequential-oracle gate through the device path | 17/17, continuation identical |
| anchor A/A | 7.386 ms, spread 0.015 (third box within 1.5% of the shipped config) |

## The three-way at K=16 — quantifying the singleton artifact

| arm | verify (17 rows) | basis |
|---|---|---|
| eager grouped | 82.99 ms | eager loop (host sync; not capturable) |
| singleton, captured | 47.24 ms | graph (matches box 8's 47.71 — cross-box) |
| **device grouped, captured** | **36.38 ms** | graph |

The reuse-disabled artifact inside the S2 bound was **10.86 ms (23%)**.
Grouping works — rows sharing an expert now share its packed-weight
read, with bitwise-identical outputs — and the verify step is still
**4.93× the anchor**.

## The verdict numbers

| K | accept (committed) | verify_graph_ms | T_pred | tok/s |
|---|---|---|---|---|
| 16 | 2.948 | 36.38 | 0.60× | 81.0 |
| 32 | 3.447 | 39.14 | 0.65× | 88.1 |
| 64 | 3.926 | 43.72 | 0.66× | 89.8 |

Speculation LOSES ~34% at every measured K, with the best available
reuse mechanism, on receipts. The verify cost curve is remarkably
flat in K (36.4 → 43.7 ms for 17 → 65 rows): the cost is dominated by
per-window fixed structure (the distinct-expert union's weight
traffic plus the T>1 path's own overheads), exactly the shape the S2
overlap data predicted — the union grows slower than rows, but its
absolute price already dwarfs the accepted-token yield.

## The composed ceiling and the 425 statement (licensed by the map)

Per the registered adjudication, a REFUTED grouped receipt is the
condition for recomputing the composed single-stream ceiling:

- **Speculative decoding: closed by direct measurement.** 425 via
  speculation would need verify(33 rows) ≤ 8.08 ms against a 7.39 ms
  anchor; measured grouped verify is 39.1 ms. No step-cheapening
  rescues a mechanism that loses throughput outright at every K.
- **Step-cheapening lanes** (all receipts): certified 135.4 tok/s;
  fusion tail ~150; K6's dot-pad observed at 46.4 µs/pair — the
  PARTIAL band of its pre-set timing bars — pending its correctness-
  bar amendment (~177 trajectory if it certifies); the K4 loads-floor
  trajectory ~294 remains the lane's ceiling if a bespoke kernel ever
  fully lands.

**425 tok/s single-stream is REFUTED for this model on this
hardware** — this time with the grouped receipt the question
required: verification cost was measured on the best available reuse
mechanism, with bitwise parity gates, and the crossing mechanism
loses money at every window size. The realistic composed ceiling is
**~150–290 tok/s** depending on how far the kernel lane's remaining
registered work lands.

## What survives this refutation

The device-grouping machinery (bitwise-equal to eager grouping,
capture-safe, CI-gated) is now product-grade infrastructure: any
future T>1 captured path — batch>1 decode graphs, chunked-prefill
capture, shared-expert models where the union IS small — gets it for
free.
