# RESULTS — G8 B=16: NOT-CLOSED, and the imbalance has flipped sides

Scored under [PREREG-g8close.md](PREREG-g8close.md) by the committed
[g8close_verdict.py](g8close_verdict.py); receipts in
[receipts-g8g9/](receipts-g8g9/). Box: EPYC 9655 + RTX 5090, swap
self-test OK, static arms bit-identical, A/A balance spread 0.005.
Cycle (shared with the G9 passes) ≈ $1.10, box destroyed, zero
instances.

## Verdict

**NOT-CLOSED** — best serving arm 0.497 vs the 0.80 gate, far beyond
noise.

| arm | balance | gpu ms/step | dram ms/step | dram wall |
|---|---|---|---|---|
| A1 static | 0.495 | 29.55 | 14.61 | 20.1 s |
| A2 static | 0.499 | 29.50 | 14.73 | 20.4 s |
| C controller | 0.384 | 29.46 | **11.31** | **15.6 s** |
| R their-calib | 0.493 | 29.53 | 14.56 | 19.9 s |

## What the receipts actually say

1. **The imbalance flipped.** The g8-era pair was gpu 15.7 / dram 22.9
   (CPU-limited, 0.688). On the current stack it is gpu **29.5** /
   dram **14.6** — the GPU expert bus is now the tall pole, so no
   CPU-side improvement can close a min/max balance gate. The
   registered honesty clause did its work: the controller arm is the
   BEST arm on the wall (−22%) and the WORST on balance (0.384).
2. **Their calibration alone does not reproduce the g8 placement.**
   The R arm's solver output matches ours (~14.6 ms dram), not the
   layer-concentrated 22.9 ms shape — that shape needed their routing
   profile, not just their cost constants. The 0.688 starting point was
   partly an artifact of its own placement.
3. Gate disposition per the hard stop: OPEN, no re-run. Closing
   min/max balance on this stack means GPU-expert-bus work (29.5
   ms/step for ~2,000 VRAM uniques ≈ 15 µs/unique H2D+GEMV — its own
   line if wanted), or conceding that the campaign's balance metric no
   longer tracks the spec's objective (bytes-through-DRAM and wall),
   on which the controller arm is simply the best configuration
   measured. Both readings are recorded; neither is scored.
