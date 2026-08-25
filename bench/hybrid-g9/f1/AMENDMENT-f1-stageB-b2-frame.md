# AMENDMENT — B2's bars were absolute; the claim is relative

Registered 2026-08-25, after the B2 arms completed and the calculator
REFUSED, and before any re-adjudication.

## What happened

The B2 arms ran clean on box 5 (receipts `receipts-f1/stageB-b2/`):
bitwise gate 13/13 on the box's GPU, C0 = 9.589/9.575 ms (A/A
0.014 ms), C1 = 7.497 ms, token identity exact, 0 recompiles. The
calculator returned:

```
REFUSE: base 9.57 ms is already at or under the PASS bar 9.6 --
the frame is degenerate and no gain is measurable
```

That refusal is CORRECT as registered, and what it caught is a
registration defect: AMENDMENT-b2 set ABSOLUTE bars (PASS ≤ 9.6 ms)
derived from box 4's base (10.62 ms). Box 5 runs the identical shipped
config at 9.57 ms — a 1.05 ms cross-box shift on unchanged code
(driver 580.119 vs 595.71; the graph basis is host-insensitive, not
driver-insensitive). An absolute bar does not transfer across boxes
when the treatment's claim is relative — "remove ≥ 1.0 ms of the
1.58 ms block" — which is exactly how the amendment's own bars
section justified the 9.6 number ("≥ 1.0 ms removed").

## Amended bars (the same registered quantities, frame-invariant)

Adjudicated on the SAME receipts — no new measurement:

- **PASS**: within-box gain ≥ 1.0 ms (the census floor the original
  bar encoded), with A/A spread < half the gain bar (0.5 ms).
- **PARTIAL**: gain 0.5–1.0 ms, ships only if spread < gain/2.
- **REFUTED**: gain < 0.5 ms — the kernel does not ship.

All other refusals (bitwise gate, token identity, capture neutrality,
recompiles) unchanged and already discharged by the receipts.

`f1_verdict.py` gains a gain-frame mode (`pass_gain_ms` /
`partial_gain_ms`) that is mutually exclusive with the absolute bars;
self-tested both directions including the degenerate case that
triggered this amendment (a fast base that the absolute frame refused
must adjudicate cleanly in the gain frame).

## The cross-box lesson, recorded

Absolute step-time bars belong only to claims about absolute step
time. A treatment whose registered mechanism is "remove a block"
must be barred on the within-box gain, or a faster box turns a
better-than-expected result into a refusal — and a slower box would
have turned a worse one into a pass.
