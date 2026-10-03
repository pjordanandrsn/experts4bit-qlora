# TC1 amendment 11 read (#945): the steady-state comparison after the sync fix — P18 UNTESTED (an unstable host)

**Box:** `tc1-5090-40` (instance 53999231, Intel Xeon E5-2696 v4 @ 2.20 GHz, RTX 5090, driver 595.91.07, $1.50; receipts in
[`receipts/tc1-5090-40/`](receipts/tc1-5090-40/)). The code was e4b `23c3f9b` (#946's single-read grouping) and grouped-nf4-gemm
`133ad9d` (the pinned ring as the default). Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 11.

**What was asked.** Amendment 8's `qwen3nativebest200` token, unchanged, run against e4b after #945. P18 predicted that axolotl's
scattermoe native-best / e4b shipped over steps 101..200 would lie in [0.97, 1.15] on two stable pairs. Every arm is VALID.

## Readings

| arm | late median, steps 101..200 (s/step) | steps 11+ median | summed step time, 200 steps | held-out at 200 | J/step |
|---|---|---|---|---|---|
| e4b shipped, draw 1 | 5.922 | 5.911 | 1,182 s | 0.7936 | 1,200 |
| e4b shipped, draw 2 | 5.254 | 5.234 | 1,042 s | 0.7946 | 1,131 |
| axolotl scattermoe best, draw 1 | 5.235 | 5.285 | 1,781 s | 0.7695 | 1,557 |
| axolotl scattermoe best, draw 2 | 5.712 | 5.978 | 1,910 s | 0.7700 | 1,640 |
| e4b matched (anchor) | 6.720 | 6.957 | 1,462 s | 0.7688 | 1,522 |

**P18 UNTESTED.** e4b's two late-window draws are 12.0 % apart and axolotl's 9.1 %, both outside the registered 5 %. The drift
alternates (e4b's first draw slow, then axolotl's second), which points to the host rather than either framework. The steps are
also slow in absolute terms on this host: e4b in the same configuration stepped at 2.37 s on a Ryzen 9 7900 (`tc1-5090-42`'s
previous-body arms), and P15's Ryzen 3900X box at 3.70 s over steps 11..20.

**Read as reported, not quoted:**

- e4b again finishes a 200-step run first: 1,042–1,182 s summed against axolotl's 1,781–1,910 s, axolotl's warm-up included.
- axolotl again reaches the matched held-out curve (0.7695–0.7700 against e4b matched's 0.7688), while e4b as shipped sits
  0.025 above it.

## What follows

No register row: amendment 11's decision rules do not trigger on an UNTESTED reading, and `.native-steady-state` stays as measured
for the code before #945. Since this box ran, e4b's step has moved again (#440's trimmed LoRA delta, measured at 0.939 shipped; the
tile rule, RMSNorm and rotary changes in flight). The steady-state question is re-asked on a stable host once that queue settles.
