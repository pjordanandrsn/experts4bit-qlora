# P227 results: the per-row GEMV against K25 at 2 to 8 decode tokens (Mixtral PASS, Qwen3 MISS)

**Registration:** [PREREG-p227.md](PREREG-p227.md) (e4b#2013, with Amendments 1-4), read once, as registered.
**Reading:** `p227-5090-1`, one RTX 5090 at a 575 W power limit, e4b `b7146363` (the Amendment 4 merge), grouped-nf4-gemm
v0.50.0, $0.083. Records: [receipts/p227-5090-1/](receipts/p227-5090-1/README.md).

## The question

At T = 2 and T = 4 decode tokens, does the per-row route (arm B: `E4B_NF4_SMALLM_BW=1`, singleton groups, K33, one row per
program) run the expert call in at most 0.70 of K25's time (arm K: device grouping, then K25 at the family's served decode
plan)? Each family is ruled on its own.

## The answer

| family | T | B/K median | upper bound | clock-adjusted | rules |
|---|---|---|---|---|---|
| **Mixtral-8x7B** | 2 | 0.3565 | **0.3601** | 0.3601 | <= 0.70 |
| | 4 | 0.4608 | **0.4655** | 0.4655 | <= 0.70: **PASS** |
| | 8 (never rules) | 0.7277 | 0.7346 | 0.7346 | |
| **Qwen3-30B-A3B** | 2 | 0.7440 | **0.7478** | 0.7478 | > 0.70 |
| | 4 | 0.9002 | **0.9047** | 0.9047 | > 0.70: **MISS** |
| | 8 (never rules) | 1.1994 | 1.2121 | 1.2121 | |

- **Mixtral PASSES:** the per-row route runs the expert call in 0.36 of K25's time at two tokens and 0.47 at four.
- **Qwen3 MISSES:** 0.75 and 0.90. Lever B is small on Qwen3 at these token counts, and from eight tokens it is slower
  than K25.
- **B2's trigger fires on both families.** The per-row route over the byte floor at T = 8 is 2.00 on Mixtral (lower bound
  1.97) and 1.59 on Qwen3 (lower bound 1.57), against the 1.25 threshold.
- **All three predictions HELD** (Mixtral PASS, Qwen3 MISS, B2 fires on Mixtral).
- **The instrument is sound:** every self-pair (K2/K) median is between 0.9999 and 1.0005, and B matched K within the
  1.5e-2 tolerance everywhere (largest relative difference 0.0078).

## Reported, never ruling

| family | T | distinct experts per layer | byte floor | K / floor | B / floor | B, K (ms a replay) |
|---|---|---|---|---|---|---|
| Mixtral-8x7B | 2 | 3.53 | 1.827 ms | 3.19 | 1.14 | 2.097, 5.854 |
| | 4 | 5.50 | 2.871 ms | 2.88 | 1.33 | 3.808, 8.265 |
| | 8 | 7.23 | 3.785 ms | 2.74 | 2.00 | 7.538, 10.356 |
| Qwen3-30B-A3B | 2 | 15.08 | 1.273 ms | 1.77 | 1.32 | 1.681, 2.263 |
| | 4 | 26.45 | 2.233 ms | 1.40 | 1.25 | 2.804, 3.118 |
| | 8 | 39.49 | 3.316 ms | 1.33 | 1.59 | 5.274, 4.397 |

- A replay is the captured graph: 8 layers for Mixtral and 48 for Qwen3. The byte floor is the distinct experts' bytes
  over the copy rate measured on the box (1,519 GB/s median).
- **Why the families differ.** At T = 2 Mixtral routes 4 rows to 3.5 distinct experts a layer, and K25's 16-row tile
  spends most of its work on padding. On that card K25 runs at 3.2x the byte floor, against Qwen3's 1.8x. The per-row
  route streams each row once, so it gains most where the tile wastes most.

## What follows (as registered)

- **Mixtral's PASS opens B3:** a served read of `E4B_NF4_SMALLM_BW=1` at 2 and 4 requests. It needs its own registration
  with a quality bar, because the served tokens differ from K25's.
- **B2's trigger fired on both families,** so B2 opens: a multi-row K33 (an `M_ROWS` constexpr over device-grouped rows,
  bitwise per row against M = 1). A design note comes first, with Mixtral first, since it fired with the larger margin.
- **Qwen3's MISS sizes lever B there.** It is a finding, not a failure.
- No default changes here. `E4B_NF4_SMALLM_BW` stays opt-in.

## What this does not say

- Nothing about served speed or tokens: this is the expert call in a captured microbench, and B3 reads the served path.
- Nothing at other token counts, families, cards or power limits.
- Nothing about where the rest of the 2-8 token step goes; that is the serving roadmap's census.
