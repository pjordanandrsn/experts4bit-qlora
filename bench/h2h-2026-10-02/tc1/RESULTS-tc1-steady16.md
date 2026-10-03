# TC1 amendment 16 read: at steady state e4b as shipped now steps faster than axolotl's scattermoe on a 5090 — P27 HELD (1.238)

**Box:** `tc1-5090-46` (instance 54024744, AMD EPYC 7663, RTX 5090, driver 580.95.05, $0.99; receipts in
[`receipts/tc1-5090-46/`](receipts/tc1-5090-46/)). The code was e4b `e5859e9` and grouped-nf4-gemm `00929a4`, with every default
from amendments 10–15:

- the single-read grouping and pinned ring (#945);
- the trimmed LoRA delta (#440);
- the cost tile rule (#442);
- the fused rotary (#965);
- the fused RMSNorm (#975).

No environment variables were set. Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 16.

**What was asked.** Amendment 8's `qwen3nativebest200` token, unchanged. Over 200 field-recipe steps it runs e4b as shipped (bf16 expert
adapters, its own init) against axolotl 0.20.0's scattermoe native-best (fp32 adapters, its own init), two interleaved draws each, then
e4b's matched arm as the anchor. P27 predicted that axolotl / e4b over steps 101..200 would lie in [1.05, 1.50]. Every arm is VALID.

## Readings

| arm | late median, steps 101..200 (s/step) | steps 11+ median | summed step time, 200 steps | held-out at 200 | J/step | peak GB |
|---|---|---|---|---|---|---|
| e4b shipped, draw 1 | 3.223 | 3.212 | 654 s | 0.7948 | 827 | 25.51 |
| e4b shipped, draw 2 | 3.242 | 3.211 | 650 s | 0.7937 | 809 | 25.51 |
| axolotl scattermoe best, draw 1 | 4.014 | 4.012 | 1,373 s | 0.7705 | 1,475 | 26.77 |
| axolotl scattermoe best, draw 2 | 3.991 | 4.029 | 1,389 s | 0.7704 | 1,461 | 26.77 |
| e4b matched (anchor) | 3.662 | 3.769 | 766 s | 0.7673 | 991 | 28.29 |

**P27 HELD.** axolotl scattermoe / e4b shipped over steps 101..200 is **1.238**, and the four cross-draw ratios run 1.231–1.246 inside
[1.05, 1.50]. Both pairs are stable (0.6 % and 0.6 %). The reducer's line reads FALSIFIED against P14's own [0.90, 1.10] band; P27 reads
the same line against its band, as registered.

**Ordering: e4b shipped faster.** The whole interval lies above 1.0. P14 and P15 (0.911 and 0.901 on two hosts, the code before #945)
had axolotl about 9–10 % faster; e4b after amendments 10–15 is now about 24 % faster at steady state on this host.

Also read, not quoted as positions:

- **Whole run:** e4b finishes 200 steps in 650–654 s against axolotl's 1,373–1,389 s, about 2.1× sooner, axolotl's warm-up included.
- **Energy and memory:** axolotl spends ×1.79 the energy per step, at 1.26 GB more peak memory (26.77 against 25.51 GB).
- **Quality:** axolotl reaches the matched held-out curve (0.7704–0.7705 against e4b matched's 0.7673), while e4b as shipped sits
  0.024–0.027 above it. That is its known plateau from its own init; the comparison is reported, not judged.

## What follows

A new register row, `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state`, records the post-amendment-15 ordering.
`.native-steady-state` (2026-10-02) stays as measured, labelled as the code before #945; the two boxes' numbers are not divided.
STATUS now says e4b as shipped steps faster than axolotl's scattermoe at steady state on that host. One host has measured it so far.
