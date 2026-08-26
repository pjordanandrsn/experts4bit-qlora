# PREREG — BV3b: grouping parity + the kernel-swap identity frame

Registered 2026-08-26, before any measurement, as required by
RESULTS-bv3 before any re-adjudication of its wall receipts.

## Question

BV3's identity refusal attributed token divergence to device-vs-eager
MoE grouping numerics (pre-window divergence, no sequence crossing,
two fully-identical rows, both arms internally deterministic). Is
that attribution TRUE — i.e., is the device-grouped path's numeric
difference inside the fp-reorder class — or is there a real defect?

## Probe (on-box, instruments only)

Per layer, on IDENTICAL inputs at decode shapes (T=16 rows, real
routing from a live step): run the expert stack through BOTH paths
(eager grouping; device grouping) and record per-layer
`max|Δ|` / `max|ref|` on the MoE output. Frame: the K6 relative bar,
`max|Δ| ≤ max|ref|·2⁻⁷`, `max_abs_ref` recorded per layer. All 48
layers; REFUSE if any layer exceeds the frame (that is the defect
case and it blocks the lane).

## The re-derived identity frame (applies ONLY if parity passes)

For a treatment that SWAPS kernels, token identity vs the old kernel
is not a mechanism property (the F2-T2 lesson; BV3 receipts). The
frame becomes:
1. **Parity**: the probe above, all layers inside 2⁻⁷.
2. **Determinism**: each arm's runs token-identical to each other
   (already held in the BV3 receipts: graph_a ≡ graph_b, all rows).
3. **No crossing**: every graph row's best-match eager row is itself
   (already held).
4. **Quality**: per-row degeneracy law on every stream (already
   held), plus greedy-continuation sanity — divergent rows must
   remain fluent continuations (spot-recorded, not gated).
Under this frame the BV3 wall receipts re-adjudicate WITHOUT rerun:
the bars stay BV3's (PASS ≥1.5× fresh-eager aggregate with the frame
green and the B=1 sanity; PARTIAL ≥1.2×).

## Rider on the same box (separate deliverable)

The SV1 busy-floor census re-run (lost to the teardown-manifest
failure): profiled current-defaults decode run, Self-CUDA per replay
vs the 3.64 ms 275-target line, per the SV1 pre-commitment.

## Calculator

`bv3b_verdict.py`, self-tested; receipts under
`bench/hybrid-g9/bv3b/receipts-bv3b/`. REFUSE: any layer outside the
parity frame, missing layers, a determinism or crossing regression
in the referenced BV3 receipts.
