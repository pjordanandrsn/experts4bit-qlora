# RESULTS — slot controller C1: CONTROLLER-CERTIFIED

Scored under [PREREG-controller.md](PREREG-controller.md) by the
committed [controller_verdict.py](controller_verdict.py); receipts in
[receipts-controller/](receipts-controller/). Box: EPYC 9755 + RTX
5090, G0 PROCEED; ten FRESH half-stride-shifted windows (base 14500,
stride 28400), gen 128, per-step series; prior = the committed
design-set series ([receipts-online](receipts-online/)). NVMe empty
(asserted by the verdict, per window). Cycle ≈ $0.79, box destroyed,
zero instances.

## Verdict — the line's first stage-gate PASS

| metric | fresh (scored) | design set (sizing) | bar |
|---|---|---|---|
| raw reduction vs deployed static | **18.4%** | 19.8% | ≥ 15% |
| swap-adjusted (2 uniques/swap) | **16.0%** | 17.3% | ≥ 10% |
| swaps (10 windows) | 1,385 | 1,637 | reported |
| split-half oracle | 39.8% | 37.3% | ≥ 8% (control) |
| oracle-gap captured | 40% | 46% | reported |

Every window improved individually (w0–w9 all controller < static; the
largest single-window gain 23.4% at w3, the smallest 9.8% at w9). The
design-to-fresh transfer barely moved the numbers — the rule's value is
not an artifact of the windows it was designed on.

## What the frozen rule buys, in engine terms

At the 10 GB operating point the controller removes ~18% of the DRAM
touch-mass the deployed static profile pays. Priced at the (host-
labeled, uncertified) dense marginal ~17–26 µs/unique, that is roughly
**0.5–1.2 ms/step** of CPU decode bill on this workload — bought with
~1 swap/step ≈ 0.1 ms/step of H2D at the measured pinned bandwidth,
already netted in the adjusted metric. The noise-aware economic gate is
what makes this real: the naive controller's 27.9% raw gain was worth
2.7% after churn; the frozen rule keeps 16% net.

## C2 inheritance (per the prereg)

C1's pass authorizes registering C2 — the in-engine build. It inherits:
1. The rule VERBATIM (epoch 8, trailing-32, prior floor 0.25, gate
   `max(4/32, 3σ)`, S = 4045).
2. The swap budget: ~1 swap/step sustained, bursts ≈ 265/window at
   content shifts — the H2D machinery's sizing spec.
3. The online cycle's boundary data (73–232% persistence error at hard
   content switches): the tier swap path must tolerate a burst of
   wrong-way swaps for ~one horizon after a switch, or gate on a
   change-point detector.
4. The measurement design C2 must carry: end-to-end wall time under
   the A/A gate, with per-bracket discriminability spoilers (the
   regime cycle's lesson) — bracket sizing such that predicted savings
   exceed 3× pass spread at the worst accepted gate noise.

C2 is engine work under its own registration — the program owner's
call, not this cycle's.
