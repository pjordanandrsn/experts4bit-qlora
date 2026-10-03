# TC1 amendment 15 read (#945): e4b's fused training RMSNorm makes the step 4–8 % faster on a 5090, held-out unchanged — P24, P25 and P26 HELD

**Box:** `tc1-5090-45` (instance 54018669, AMD EPYC 7663, RTX 5090, driver 580.95.05, $0.39; receipts in
[`receipts/tc1-5090-45/`](receipts/tc1-5090-45/)). The code was e4b `aba67ea` (#961, #965) and grouped-nf4-gemm `00929a4` (#442, the cost
tile rule as default). Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 15. (`tc1-5090-44`, the first
launch, ended HARNESS_ERROR at $0.0004 before its command ran: the guard's Vast auth probe got one HTTP 429. adertha-agents#138 now
asks again with backoff.)

**What was asked.** e4b against itself at the field recipe, on the trimmed LoRA delta and the post-#945 sync path. The two sides were:

- `*_rms0`: `E4B_FUSED_RMSNORM=0`, the Hugging Face RMSNorm composite;
- `*_rms1`: `E4B_FUSED_RMSNORM=1`, #961's fused kernel, one launch each way, near-exact.

It ran on the shipped and matched arms, two draws each, in ABBA order. Every arm is VALID. Each rms1 record shows 193 norms patched
and 33,888 fused calls; each rms0 record shows none.

## Readings

| arm | side | s/step (steps 11+) d1 / d2 | held-out at N d1 / d2 | J/step d1 / d2 | first training loss |
|---|---|---|---|---|---|
| shipped | composite | 3.537 / 3.386 | 0.8143 / 0.8139 | 933 / 871 | 2.0705 |
| shipped | fused | 3.234 / 3.159 | 0.8136 / 0.8103 | 834 / 837 | 2.0614 |
| matched | composite | 4.258 / 4.396 | 0.8534 / 0.8477 | 1,089 / 1,100 | 2.0705 |
| matched | fused | 4.110 / 4.192 | 0.8482 / 0.8499 | 1,060 / 1,067 | 2.0614 |

- **P24 HELD:** fused / composite is **0.924** on the shipped arm, with the four cross-draw ratios at 0.893–0.955 (band 0.85–0.97).
- **P25 HELD:** **0.959** on the matched arm, cross-draw 0.935–0.984 (band 0.88–0.98).
- **P26 HELD:** the two sides' mean held-out at N differ by −0.0021 (shipped) and −0.0015 (matched), against a band of 0.01.
- The kernel is not bit-identical, and the difference shows before any update. The first training loss is 2.0614 against 2.0705,
  and held-out at step 0 is 1.9441 against 1.9505: a one-ulp change can flip a close router top-k choice. The trajectories then
  agree within P26's band.
- The host is slower than the earlier boxes' (shipped 3.2–3.5 s/step, against 2.0–2.2 on the Ryzen hosts), and its draws are
  noisier (within 4.3 %); every pair passed the 5 % stability gate.

## What follows

By amendment 15's decision rule, the fused RMSNorm is on by default (#975; `E4B_FUSED_RMSNORM=0` turns it off). Register:
`e4b.train.fused-rmsnorm.qwen3.5090.2026-10-03`. Amendment 16 (#972) re-asks the steady-state comparison against axolotl's
scattermoe with e4b after all of this.
