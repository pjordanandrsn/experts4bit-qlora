# P158 results: K39's decode in the served step (FLIP, g16 1.0925)

**The read** (`p158-5090-1`, one RTX 5090, 2026-10-20; receipts under [`receipts/p158-5090-1/`](receipts/p158-5090-1/)).
On the shipped default server for Qwen3-30B-A3B, `E4B_K25_LUT=prmt` (K39's decode for K25, grouped-nf4-gemm v0.50.0)
decoded 16 concurrent requests **9.2 %** faster than today's select tree: g16, the min of four ABBA pairs, is
**1.092458** against the 1.03 bar. One request was unchanged (g1 **0.999856** against the 0.99 guard): K25 is not on the
T == 1 path. Every token the prmt arms decoded is the tree's. **FLIP.**

The registration is [`PREREG-p158.md`](PREREG-p158.md) (e4b#1864), and the launch commit is its merge, `406a391a`. It ran
grouped-nf4-gemm v0.50.0 at `41ef086e`, on torch 2.8.0+cu128, triton 3.4.0 and transformers 5.17.0. The verdict was
re-derived from the private receipt store, byte-identical to the box's, and it re-derives from the public arm records
here too.

## The numbers

| | |
|---|---|
| W16 decode, tree (S0a-S0d) | 975.85-976.44 tok/s |
| W16 decode, prmt (S1a-S1d) | 1066.72-1067.28 tok/s |
| **g16**, the min of four W16 pairs (S1a/S0a ... S1d/S0d) | **1.092458** (1.0925, 1.0934, 1.0934, 1.0932) |
| **g1**, the min of four W1 pairs | **0.999856** (1.0007, 0.9999, 1.0005, 1.0006) |
| W16 self-pairs / W1 self-pairs | 0.9994-1.0003 / 0.9998-1.0006 |
| median sampled SM clock over W16, S1 vs S0 | 2842 vs 2835 MHz (+0.25 %; no pair labelled) |
| cost | $0.4190 (worst registered $2.35) |

- **Identity:** the tree decoded the same tokens in all five of its processes (the null), and the prmt arms decoded
  those tokens too (the contract), at W16 and W1, at 32 and 160 tokens.
- **Engagement:** every process's build recorded exactly 960 K25 calls under its own decode and nothing under the
  other, and exactly 480 bandwidth-GEMV calls at T == 1. Both early gates passed. The two closed forms, derived from the
  code and confirmed in part on a $0 layer-cut build (`layercut/`), hold on the 5090.

## The predictions, graded

| prediction | read | grade |
|---|---|---|
| g16 in [1.06, 1.14] | 1.092458 | HELD |
| g1 in [0.99, 1.01] | 0.999856 | HELD |
| every W16 and W1 self-pair in [0.99, 1.01] | 0.9994-1.0006 | HELD |
| every S1 token identical to S0's | identical | HELD |
| no pair clock-confounded | none | HELD |
| verdict FLIP | FLIP | HELD |

**Against the projection:** P156's microbench ratio carried 1:1 would give 1.121, and its clock-adjusted lower bound
1.077. The read, 1.092, sits between them. Taking the K25 pair at P152's census time (11.748 ms) within this read's own
tree step (16.392 ms, from the S0 arms' median 976.07 tok/s at W16), 1.092 corresponds to a K25 pair 1.134x faster in
the served step, against P156's 1.1755 in the microbench. That rests on the census time carrying over; this read does
not time the pair alone.

## Caveats

- **One card, one family, one width:** an RTX 5090 at 575 W, Qwen3-30B-A3B at 16 rows (W16) and 1 row (W1), today's K25
  plan. Other families, cards, row counts and plans are their own reads; decode x plan (P156's reported C1-C3) is a
  later lane.
- **Identity on P109's prompts:** the token gate held on the registered workloads. The kernel's general bitwise
  contract is grouped-nf4-gemm's `test_nf4_smallm_prmt_gpu.py`, which passed on this card as the premise.
- **Clocks:** sampled every 500 ms. Both arms ran under the same power-cap event (0x4) at about 420-430 W against the
  575 W limit; the 0.25 % clock difference is far below the 2 % label and the 9 % effect.

## What follows

- **A default flip, qwen3_moe only.** A separate PR may make `prmt` qwen3_moe's K25 decode by default, citing this
  record. The default must not break an install on an older kernel package. Unset, the knob resolves to `prmt` where the
  installed grouped-nf4-gemm carries K39 on NVIDIA, and to `tree` otherwise, with the reason recorded in
  `info["k25_lut"]`. An explicit `E4B_K25_LUT=prmt` keeps its hard refusal.
- **Decode x plan:** C1-C3's plans with prmt, on their own held-out read, as a later lane.
