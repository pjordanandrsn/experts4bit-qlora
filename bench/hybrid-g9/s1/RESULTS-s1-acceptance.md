# RESULTS — S1 Stage A: MARGINAL → S2-lite registered for n1_k16

Measured 2026-08-25 under PREREG-s1-acceptance. Receipts in
`receipts-s1/stageA/` (RTX 5090, driver 595.58, torch 2.13/triton 3.7,
e4b `b63b02f`, gnf4 `4f685c0`; instance destroyed, vast zero).

```
S1 VERDICT: MARGINAL
  best conservative 0.31x (cell n1_k16), best optimistic 2.95x
  -- register S2-lite for the single best cell only
```

## The two numbers

**Acceptance is strong.** Over 3840 windows from 30 non-degenerate
traces (2 of 32 refused: one 20-distinct-token trace, one 8-gram
looping 8× — the degeneracy gate earning its place), prompt-lookup at
n_min=1, K=16 yields **2.948 tokens per verify step**. The
distribution is bimodal exactly as the mechanism predicts: 62.9% of
windows draft nothing usable (m=0), but 5.2% accept the FULL 16-token
draft, and the long tail pays for the whole scheme. Every cell in the
3×3 grid clears the 1.1× optimistic floor — the drafter is not the
problem.

**The measured verify path is not viable.** The prefill-continuation
probe prices a (K+1)-row verify at 67.6–70.2 ms against a 7.406 ms
anchor (A/A 0.006 ms; 135.0 tok/s — the shipped config reproducing
within 1.3% of box 5). Two known components, both disclosed in the
prereg: eager host submission (the b1d receipts put the eager T=1
step at ~51 ms vs 15.2 graphed — the host gap alone explains most of
it) and the prefill expert path. The bound is sound and the verdict
respects it: conservative 0.31× forbids GO.

## What MARGINAL registers (per the prereg, unchanged)

**S2-lite**: a graph-compatible executor for the single cell n1_k16 —
draft 16 tokens by prompt-lookup (host-side, zero device cost),
verify all 17 positions in ONE graphed step, accept the longest exact
prefix. Its economics from THIS receipt: 2.948 tokens/step against a
graphed 17-row verify that costs somewhere between 1.0× and ~2.3×
the anchor (the eager probe's 9.4× is host gap, not device work —
the b1d eager/graph ratio bounds the device-side inflation). S2-lite's
bars will be set as gain bars against its own measured graphed verify
step, with greedy token identity as the hard refusal.

## Notes for the record

- The verify probe's numbers are per the registered instrument:
  UPPER bounds through an executable path. They are loose because
  eager, not wrong. S2-lite's first arm is precisely "graph the
  verify step and measure it" — the number this cycle deliberately
  did not fabricate.
- Anchor 7.406 ms = **135.0 tok/s**, second box in a row to reproduce
  the shipped config within 1.5%.
