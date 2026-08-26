# RESULTS — BV3: the B>1 CUDA-graph decode loop

Adjudicated 2026-08-26 by `bv3_verdict.py` on
`receipts-bv3/bv3_report.json`. Box: instance 48709950 (post-F2
serving anchor 7.25 ms).

## VERDICT: REFUSE — identity gate, as registered

```
eager B=16: 130.8 / 132.4 ms/step        graph B=16: 38.00 / 37.99 ms
(would-be ratio 3.44x; 122 -> 421 aggregate tok/s)
graph B=1 sanity: 7.25 ms (== the box's own current-defaults class)
graph_a == graph_b: token-identical, all 16 rows; eager pair likewise
REFUSE: row 11 diverges from its eager stream at step 3 (< 32)
```

The registered gate refused, and the refusal stands. The wall
numbers are REAL and reproducible (A/A spread 0.01 ms, zero
recompiles) but UNCERTIFIED: the prereg's identity frame required
graph-vs-eager agreement, and it was not met.

## What the receipts attribute the divergence to

- Every graph row best-matches ITS OWN eager row (no sequence
  crossing); rows 0 and 2 are identical for all 128 steps.
- Several rows diverge INSIDE the pre-window — during the graph
  run's ordinary scheduler phase, before any capture — where the
  only difference from the eager arm is the MoE grouping path
  (device grouping, which the treatment enables at T>1).
- Both arms are internally deterministic (run-to-run
  token-identical).

Diagnosis: device-grouped and eager-grouped MoE are two valid
kernel paths with different accumulation orders; the 128-expert
router amplifies low-bit logit differences at near-ties into early
token divergence. The identity gate borrowed the same-kernel
K6-B frame across a kernel swap — the F2-T2 bitwise-claim lesson
repeated at the batch level, this time caught by the registered gate
rather than before registration.

## Registered next step (BV3b — before any re-adjudication)

1. An on-box logit-parity probe: device vs eager grouping on
   identical decode-shape inputs (T=16 rows), the K6 relative frame
   (`max|Δ| ≤ max|ref|·2⁻⁷`, `max_abs_ref` recorded), per layer.
2. If parity is inside the reorder class, BV3b registers a
   re-derived identity frame for kernel-swap treatments (parity
   bound + per-arm determinism + no sequence crossing) and
   re-adjudicates against the SAME wall receipts; if parity is
   outside the class, the divergence is a real defect and the lane
   is blocked on it.
3. The SV1 busy-floor census rides the same box (its kernel table
   was lost to a teardown-manifest failure, recorded in the SV1
   RESULTS).

No wall number from this cycle is citable until BV3b adjudicates.
