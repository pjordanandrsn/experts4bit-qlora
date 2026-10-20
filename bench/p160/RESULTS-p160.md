# P160 results: decode x plan (GO, C1 1.6598x over the served default)

**The read** (`p160-5090-2`, one RTX 5090, 2026-10-20; receipts under [`receipts/p160-5090-2/`](receipts/p160-5090-2/)).
K39's decode at P156's first carried plan pair, C1 (gate_up 64/256/4/2, down 128/128/8/3), ran both W16 K25 GEMMs
**1.6598x** faster than prmt at today's plan, which is qwen3_moe's served default since P158. That is the median of 126
paired reps on S1, a router-replay stream no K25 microbench had timed, with a lower bound of **1.6568** and a
clock-adjusted bound of **1.6525** against the 1.05 bar: **GO.** Every arm kept the tree's bits on every rep. This is a
microbench premise: it opens a served read and quotes no gain into the served step.

The registration is [`PREREG-p160.md`](PREREG-p160.md) (e4b#1870), and the launch commit is its merge, `2267635d`. It ran
grouped-nf4-gemm v0.50.0 at `41ef086e`, on torch 2.8.0+cu128 and triton 3.4.0. The verdict was re-derived from the
private receipt store, byte-identical to the box's, and it re-derives from the public bench record here too.
Amendment 1 (e4b#1874, hosts with no recorded power cap) was never needed: a recorded host appeared within the launch
loop, and it was closed unused.

## The numbers

| | |
|---|---|
| P, prmt at today's plan, both GEMMs a step (S1) | 10.348 ms |
| **C1** (S1) | **6.231 ms** |
| **C1's median P/C1** on S1 (SE, lower bound, clock-adjusted) | **1.6598** (0.0015, 1.6568, 1.6525) |
| C2 / C3 over P (S1) | 1.6100 / 1.6304 |
| T/P, the decode alone (S1) | 1.1660 (P156: 1.1755) |
| self-pair P/P2 (S1) | 0.9999 |
| C1 on S1's least-overlapping half (steps 64-125) / without P156's selection step numbers | 1.6557 / 1.6594 |
| C1 on S0R, P156's held-out steps 64-123 | 1.6508 (P156's reported figure there: 1.6167) |
| cost | $0.0710 (worst registered $1.16) |

- **Identity:** every arm matched T's bits, and P2 matched P's, on all 186 reps of both blocks.
- **Clocks:** no arm is labelled. C1's median SM clock is 1.0026 of P's and its memory clock is identical (13801 MHz in
  every arm). The software power-cap event is in the same share of every arm's samples (0.894-0.897 on S1), and the
  enforced limit read 575 W in every sample.
- **Compiled on sm_120:** C1's gate_up at 194 registers and 17,408 B of shared memory, its down at 127 and 26,624 B,
  against P's 64 and 6,400 B; no arm spills.

## The predictions, graded

| prediction | read | grade |
|---|---|---|
| C1's median P/C1 on S1 in [1.50, 1.70] | 1.6598 | HELD |
| C1 on S1 steps 64-125 within 0.03 of C1 on all of S1 | 1.6557 against 1.6598 | HELD |
| C1 on S0R within 0.03 of C1 on S1 | 1.6508 against 1.6598 | HELD |
| T/P on S1 in [1.15, 1.20] | 1.1660 | HELD |
| the self-pair within 0.005 of 1 | 0.9999 | HELD |
| C1's median SM clock within 2 % of P's | 1.0026 | HELD |
| every arm keeps T's bits on every rep | all of them | HELD |
| C1 compiles to 194 (gate_up) and 127 (down) registers, no spills | as predicted | HELD |

**Against P156:** on P156's own held-out steps, under this lane's spacer, order and clocks, C1 reads 1.6508 against
P156's reported 1.6167. P156 ran on another RTX 5090 (machine 40959) and this read on machine 147099, so the 2 % between
the two figures spans two cards and is not attributed. What holds is narrower: on this card, the spacer, the order and
the per-arm clocks did not lower C1 below P156's reported figure. P156's winner's curse (selection 1.770 against
confirmation 1.617) was between its selection and its confirmation; this read does not move it.

## Caveats

- **Microbench only:** the two K25 GEMMs alone, with no attention, router or glue. bf16 teacher-forced picks, not the
  served NF4 picks. One RTX 5090 at 575 W, Qwen3-30B-A3B's shapes at 16 rows, triton 3.4.0.
- **S1 shares S0's prompts:** 43.3 % of its (row, step) cells match S0's at every layer, front-loaded (PREREG, "The
  data"). The split on its least-overlapping half (31.0 % overlap) reads 1.6557, within the predicted 0.03.
- **A plan is its own compile:** C1 and P are different cubins, so the ratio prices C1's whole compile. That is what a
  served change would ship. No stage of it is attributed.

## What follows

- **An e4b PR:** the served route has one K25 plan for both projections (`_K25_PLAN`). C1 needs a per-projection plan
  knob, opt-in (`E4B_K25_PLAN`, `today` or `c1`), tied to the prmt decode (the tree runs these plans slower, P155), with
  a bitwise check that tree, prmt and prmt at C1's plans agree through the served call. It carries no speed claim.
- **A served read,** its own registration on P158's harness: prmt at C1's plans against today's served default, with its
  own bars. Any projection from this microbench enters it as a graded prediction, never a claim.
