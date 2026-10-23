# P229 results: the int8 lm_head at 2, 4 and 8 rows on K37's released `linear_i8` (GO_ROWS_8)

**Registration:** [PREREG-p229.md](PREREG-p229.md) (e4b#2026, with Amendments 1-6), read once, as registered.
**Reading:** `p229-5090-3`, one RTX 5090 at a 575 W power limit, e4b `32fda64d` (the Amendment 6 merge), grouped-nf4-gemm
v0.50.0, $0.061. Records: [receipts/p229-5090-3/](receipts/p229-5090-3/README.md).

## The question

At M = 2, 4 and 8 rows of Qwen3-30B-A3B's 151936 x 2048 lm_head on its per-row int8 grid, does the route
(`E4B_INT8_HEAD_ROWS`, arm I: K37's `dense_i8.linear_i8` at its shipped plan) run in at most 0.70 of the served path's time
(arm D: `F.linear` on the grid's bf16 rounding), and within 1.25x of the head's one-row time (arm K: K36 over the same int8
bytes)?

## The answer

| M | I/D median | I/D upper bound | I/K median | I/K upper bound | rules |
|---|---|---|---|---|---|
| 2 | 0.5109 | **0.5109** | 1.0217 | **1.0217** | PASS |
| 4 | 0.5136 | **0.5148** | 1.0272 | **1.0296** | PASS |
| 8 | 0.5081 | **0.5085** | 1.0108 | **1.0116** | PASS |

- **GO_ROWS_8.** Every read M up to 8 passes both bars on the upper bound and on its clock-adjusted value (equal here: no
  arm ran a faster SM clock). `E4B_INT8_HEAD_ROWS` is licensed up to 8, its registered maximum.
- At every M the route runs the head in about half the served path's time: a median 0.51 of D's.
- At every M it stays within 3 % of the one-row int8 GEMV: a median 1.01 to 1.03 of K's.
- **The instrument is sound.** The installed route was as registered (`rows_max` 8, K37 at plan (32, 256, 4) at every M from
  2 to 8). So was the engagement: K36 at one row, K37 at 2, 4 and 8. I matched D within the 1.5e-2 tolerance in every rep
  (largest relative difference 0.0078). The self-pair D/D2 had a median of 1.0000 at M = 2 and 4 and 1.0054 at M = 8,
  inside the 0.99-1.01 band.

## Predictions

| prediction | read | graded |
|---|---|---|
| GO_ROWS_8 | GO_ROWS_8 | **HELD** |
| I/K about 1.03-1.04 | medians 1.0217, 1.0272, 1.0108 | **MISSED**, on the cheap side: the multi-row tile costs less over the one-row GEMV than predicted |
| the self-pair within 0.005 of 1 | medians 1.0000, 1.0000, 1.0054 | **MISSED at M = 8** by 0.0004 (upper bound 1.0078); held at 2 and 4 |

## Reported, never ruling

| M | D, D2, I, K (ms a call) | D/K | I / int8 byte floor | D / bf16 byte floor |
|---|---|---|---|---|
| 2 | 0.3768, 0.3768, 0.1925, 0.1884 | 2.00 | 0.94 | 0.92 |
| 4 | 0.3768, 0.3768, 0.1935, 0.1884 | 2.00 | 0.94 | 0.92 |
| 8 | 0.3789, 0.3768, 0.1925, 0.1905 | 1.99 | 0.94 | 0.92 |

- **The byte floors:** the head's bytes over the copy rate measured on the box (1,517 GB/s). That is 0.2055 ms for the
  311 MB of int8 and 0.4101 ms for the 622 MB of bf16. Both arms run slightly under their floor, so at this shape the
  copy-rate floor is a conservative reference, not a bound.
- **The timer:** each timed replay reads on a 2.048 µs grid. Many reps tie exactly, which is why I/D at M = 2 has an SE
  of 0, and why D/D2 at M = 8 is one grid step from 1.
- **The clocks:** SM clock medians 2,865-2,872 MHz across all twelve arm-and-M cells, no cell clock-confounded. Power
  medians 238-334 W under the 575 W limit enforced in every sample.

## What follows (as registered)

- **No cap change.** GO_ROWS_8 licenses the route's registered maximum, so `MAX_ROWS` stays 8.
  - The unread row counts (3, 5, 6, 7) are licensed with their neighbours: K37 takes M as a runtime argument inside one
    16-row tile, at the same plan for every M from 2 to 8.
- **The served read is registered next:** W2, W4 and W8, ABBA on the default Qwen3 server, the knob against unset, with
  P153's quality bars per bucket. It comes before any default.
- **No default changes here.** `E4B_INT8_HEAD_ROWS` stays opt-in, and unset is 1.

## What this does not say

- Nothing about step time or served speed: this is the head alone, in a captured microbench.
- Nothing about quality: I differs from D within the tolerance, and the tokens are the served read's to judge.
- Nothing about another plan, another card, another power limit or another grouped-nf4-gemm release. The head is random
  bf16 at Qwen3-30B-A3B's shape, and the inputs are random.
- Nothing about speculative decode: the route stays refused with `E4B_PAGED_SPEC` until a read covers the pair.
