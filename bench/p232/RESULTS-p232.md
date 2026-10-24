# P232 results: the int8 lm_head at 2, 4 and 8 rows, served (FLIP_ROWS_8)

**Registration:** [PREREG-p232.md](PREREG-p232.md) (e4b#2071, with Amendments 1-2), read once, as registered.
**Reading:** `p232-5090-1`, one RTX 5090 at a 575 W power limit, e4b `f7589a02` (the Amendment 2 merge), grouped-nf4-gemm
v0.50.0, $1.00. Records: [receipts/p232-5090-1/](receipts/p232-5090-1/README.md).

## The question

On Qwen3-30B-A3B's default server, with the lm_head on its per-row int8 grid (`E4B_INT8_HEAD_BW`, on by default), does
routing the head's 2- to 8-row calls through grouped-nf4-gemm K37's `linear_i8` (arm K: `E4B_INT8_HEAD_ROWS=8`) instead of
`F.linear` on the grid's bf16 rounding (arm R: unset) keep quality and speed decode, at each group size T in 2, 4 and 8?
Per T: |mean NLL(K) - mean NLL(R)| <= 0.02 and argmax agreement >= 0.95 (P153's bars), and g_T >= 1.01, where g_T is the
smallest of the four ABBA K/R pairs at the workload with T requests.

## The answer

| T | NLL gap (K - R) | argmax agreement | KL (K against R) | g_T (min of four pairs) | rules |
|---|---|---|---|---|---|
| 2 | -0.000091 | 3064 / 3072 (0.9974) | 0.000142 | **1.0311** | PASS |
| 4 | -0.000091 | 3064 / 3072 (0.9974) | 0.000142 | **1.0243** | PASS |
| 8 | -0.000031 | 3066 / 3072 (0.9980) | 0.000142 | **1.0165** | PASS |

- **FLIP_ROWS_8.** Every T passes all three bars, so the largest T with every T up to it passing is 8.
- **TRIPWIRE none.** At every T the gap is under 0.005, the agreement over 0.98, and the KL under max(3 x the R
  self-pair floor, 0.002); the floor read 0 (R's repeat pass was identical), so the 0.002 bound applied.
- **The instrument held:**
  - every record ok, the warm-up's included; the pins (e4b, grouped-nf4-gemm v0.50.0, torch, the model) and the card
    (cc 12.0, 170 SMs) as registered; the census and fusion modes at the default; the split cap at 16 with its plans;
    every bucket 1-16 captured; the dense attention GEMV off;
  - the head's engagement: both arms swap the one 151936 x 2048 head; K routes rows 2, 4 and 8 to K37 at plan
    (32, 256, 4) and passes none; R passes them to `F.linear` and routes none; both route one row to K36;
  - 575 W enforced in every timed block of every arm;
  - every self-pair inside the NOISY band [0.985, 1.015] (0.9925 to 1.0033), and no arm's tokens differ from its own
    arm's first record (no FUNCTION_FAIL).
- **T = 2 and T = 4 read the same quality numbers,** to every recorded digit, R against B16 included. Each row's
  log-probs are the same at either group size: the bucket-stability record shows every row identical (max difference 0)
  at T = 2, 4 and 8 for both arms, so splitting the same 24 windows into groups of 2 or 4 scores the same values.
  T = 8 differs slightly.

## Predictions

| prediction | read | graded |
|---|---|---|
| quality at every T: \|NLL gap\| <= 0.001 nats and argmax >= 0.99 | gaps 0.000091, 0.000091, 0.000031; argmax 0.9974, 0.9974, 0.9980 | **HELD** |
| g_T in [1.02, 1.05] at every T | 1.0311, 1.0243, 1.0165 | **HELD at T = 2 and 4; MISSED at T = 8**, on the low side: the 8-row step gains less than projected |
| every self-pair within [0.99, 1.01] | 0.9925 to 1.0033 | **HELD** |
| at W2, most rows identical to R's at 32 tokens, and divergence by 160 | 4 of 8 W2 rows identical at 32 tokens and at 160; the others diverge at step 25 | **MISSED**: half, not most, and the divergence comes by step 25 |
| verdict FLIP_ROWS_8 | FLIP_ROWS_8 | **HELD** |

## Reported, never ruling

| workload | K/R pairs (a, b, c, d) | geomean | R decode (tok/s, Ra) | K decode (tok/s, Ka) |
|---|---|---|---|---|
| W2 | 1.0311, 1.0320, 1.0335, 1.0324 | 1.0323 | 320.95 | 330.92 |
| W4 | 1.0243, 1.0315, 1.0269, 1.0261 | 1.0272 | 545.75 | 558.99 |
| W8 | 1.0165, 1.0239, 1.0240, 1.0235 | 1.0220 | 980.53 | 996.75 |

- **The self-pairs:** W2 0.9981-1.0000, W4 0.9925-0.9995, W8 0.9961-1.0033. Across the two ABBA blocks (Rc/Ra, Kc/Ka):
  0.9955-1.0037.
- **The warm-up** (R, process 0, never scored) decoded within 0.25 % of Ra at every workload (W2 1.0021, W4 1.0016, W8
  0.9999 of Ra).
- **Greedy tokens, K against R** (reported only; K's tokens are never held to R's): 76 of 112 rows identical over
  every pair, workload and length. Where a row diverges, it diverges at the same step in all four pairs: step 25 at W2
  and W4, step 7 for one W8 row, and later steps (90, 108, 157) only at 160 tokens. Both arms decode on one int8 grid;
  K37 and `F.linear` round differently within bf16's output rounding, so near-ties can flip.
- **Against B16** (the bf16 head, quality only, reported beside P153's bars): R against B16 reads a gap of 0.0012, argmax
  0.9909 and KL 0.00064 at T = 2 and 4 (0.0011, 0.9902, 0.00065 at T = 8); K against B16's gap is 0.0011 at T = 2 and 4
  (0.0011 at T = 8). K-against-B16 argmax and KL have no instrument (Amendment 1).
- **The card:** a 575 W limit enforced in every sample; on Ra's blocks the SM clock read at least 2,812 MHz (its maximum
  setting 3,105 MHz) and the card at most 63 C.

## What follows (as registered)

- **A default PR may follow.** The verdict is FLIP_ROWS_8 and the TRIPWIRE did not fire, so a separate code PR may make
  `E4B_INT8_HEAD_ROWS=8` qwen3_moe's default, citing this read and P229. Every other model family keeps its current
  default, and the knob stays settable (unset or `1` keeps the one-row-only route).
- **No default changes here.**

## What this does not say

- Nothing about another model, head shape, card, power limit or grouped-nf4-gemm release: one RTX 5090 at 575 W, Qwen3-30B-A3B
  at its pinned revision, K37 at its shipped plan.
- Nothing beyond 8 rows: 9 to 16 rows keep `F.linear` under either arm.
- Nothing about speculative decode: the route stays refused with `E4B_PAGED_SPEC` until a read covers the pair.
- Quality is P153's instrument on 24 wikitext windows x 128 positions per T, not a downstream task.
