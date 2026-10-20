# P161 results: K25 at plan c1 in the served step, under K39's decode (FLIP, g16 1.4654)

**The read** (`p161-5090-1`, one RTX 5090, 2026-10-20; receipts under [`receipts/p161-5090-1/`](receipts/p161-5090-1/)).
On the shipped default server for Qwen3-30B-A3B, `E4B_K25_PLAN=c1` (P156's first carried tile pair, which P160 read in
a microbench) decoded 16 concurrent requests **46.5 %** faster than today's plan, both under K39's decode (`prmt`,
qwen3_moe's default since P158). g16, the min of four ABBA pairs, is **1.465433** against the 1.03 bar. One request was
unchanged (g1 **0.996642** against the 0.99 guard): K25 is not on the T == 1 path. Every token the c1 arms decoded is
today's plan's. **FLIP.**

The registration is [`PREREG-p161.md`](PREREG-p161.md) (e4b#1904), and the launch commit is its merge, `da3eb5c5`. It
ran grouped-nf4-gemm v0.50.0 at `41ef086e`, on torch 2.8.0+cu128, triton 3.4.0 and transformers 5.17.0. The verdict was
re-derived from the private receipt store, byte-identical to the box's, and it re-derives from the public arm records
here too.

## The numbers

| | |
|---|---|
| W16 decode, today's plan (S0a-S0d) | 1036.83-1039.14 tok/s |
| W16 decode, c1 (S1a-S1d) | 1522.79-1528.55 tok/s |
| **g16**, the min of four W16 pairs (S1a/S0a ... S1d/S0d) | **1.465433** (1.4721, 1.4697, 1.4654, 1.4693) |
| **g1**, the min of four W1 pairs | **0.996642** (0.9966, 0.9995, 1.0014, 0.9992) |
| W16 self-pairs / W1 self-pairs | 0.9969-1.0020 / 0.9981-1.0010 |
| median sampled SM clock over W16, S1 vs S0 | 2850 vs 2850 MHz (no pair labelled); power-cap event in 0.17-0.22 vs 0.00 of samples |
| cost | $0.726 (worst registered $2.35) |

- **Identity:** today's plan decoded the same tokens in all five of its processes (the null), and the c1 arms decoded
  those tokens too (the contract), at W16 and W1, at 32 and 160 tokens.
- **Engagement, three witnesses:** every process's build recorded exactly 960 K25 calls under qwen3_moe's default decode
  (`prmt`, env unset, `default-allowlisted`), exactly 960 under its own plan and nothing under the other, and its kernel
  received exactly 1,920 calls at its plan's block shapes (today: 32/64/4/3 for both projections; c1: gate_up 64/256/4/2
  and down 128/128/8/3, 960 each). The bandwidth GEMV read 480 at T == 1. Both early gates passed. The closed forms,
  derived from the code and confirmed in their licence half on a $0 layer-cut build (`layercut/`), hold on the 5090.

## The predictions, graded

| prediction | read | grade |
|---|---|---|
| g16 in [1.20, 1.42] | 1.465433 | **MISSED (high)** |
| g1 in [0.99, 1.01] | 0.996642 | HELD |
| every W16 and W1 self-pair in [0.99, 1.01] | 0.9969-1.0020 | HELD |
| every S1 token identical to S0's | identical | HELD |
| no pair clock-confounded | none | HELD |
| verdict FLIP | FLIP | HELD |

**The band was missed high.** It is graded, never ruling: the verdict is the registered rule's, and the band was a
prediction.

**Against the projection:** the PREREG projected g16 1.378 at a 1:1 carry and 1.280 at P158's carry. It took P158's
prmt step (14.995 ms, from its S1 arms' median 1067.03 tok/s at W16) and the K25 pair P158's saving implies (10.351 ms),
and applied P160's 1.6598, a 4.115 ms saving. This read's own medians:
- today's plan: 1038.44 tok/s, a **15.408 ms** step, 1.0275 of the step the projection assumed (the same server and
  decode, on another host);
- c1: 1524.84 tok/s, a **10.493 ms** step;
- the saving is **4.915 ms**, 1.194 of the 4.115 ms the projection implied.

That carry is reported, not attributed. This read does not time the K25 pair alone, and its host differs from P158's,
so the pair time the projection assumed is not this host's.

## Caveats

- **One card, one family, one width:** an RTX 5090 at 575 W, Qwen3-30B-A3B at 16 rows (W16) and 1 row (W1), plan c1
  against today's plan, both under K39's decode. Other families, cards, row counts, decodes and plans are their own
  reads. C2 and C3 (P160's 1.6100 and 1.6304) are not read. c1 under the select-tree decode is refused by the knob,
  never run.
- **Identity on P109's prompts:** the token gate held on the registered workloads. The general bitwise contract is e4b's
  `tests/test_k25_lut_gpu.py` with c1's cases (frozen as this lane's premise) and grouped-nf4-gemm's
  `test_nf4_smallm_prmt_gpu.py`, which passed on this card (62 premise tests).
- **Clocks:** sampled every 500 ms, the enforced limit 575 W in every sample, and the median SM clock 2850 MHz in every
  scored arm. The software power-cap event (0x4) appears in 0.17-0.22 of each c1 arm's W16 samples and in none of
  today's plan's: c1 draws more power and touches the cap, at an unchanged limit and an unchanged median clock. If
  anything that works against c1. It is part of the served difference this read measures, not a confound.

## What follows

- **A default change, qwen3_moe only.** Under the PREREG's rule a separate PR may make `c1` qwen3_moe's K25 plan by
  default, citing this record and P160. It is one read on one card, and `E4B_K25_PLAN=today` keeps the current plan
  reachable.
