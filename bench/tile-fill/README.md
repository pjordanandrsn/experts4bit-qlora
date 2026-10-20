# Tile fill and its amortization: rows per step are the lever; shared experts add a few percent (reported only)

**The construction was committed and pushed before any computation:** [`CONSTRUCTION.md`](CONSTRUCTION.md), `9e07e3b6`.
- **The first width composition failed its positive control.** Uniform (request, step) pairs read D 55.90 at B = 16,
  against e4b#1869's 57.26.
- **So it is VOID,** kept and reported, and was amended (Amendment 1, `1c6edf9f`, pushed before the amended
  computation) to a stratified composition.

**Data:** P153's router replay (`bench/p153/router-replay/rr.json.pt`, `3c59e72f`), S0 (primary) and S1. There are
1,000 draws per cell, from one generator per stream seeded at 20261020. The intervals are the draws' 2.5-97.5 %, i.e.
the spread of compositions.

**The premise, still inference:** K25 decodes an expert's NF4 block once per non-empty 16-row tile, so expert cost
tracks tiles T while useful work tracks rows. **The positive control's exact expectation** (from the reviewer's
closed form) is 57.256 (S0) and 57.419 (S1). The stratified B = 16 cells read 57.07 and 57.44, so the sampler is
unbiased, and S0's 0.19 gap is sampling (about 2.7 SE), inside the 0.2 tolerance.

## The read (S0; S1's ratios and fills within 0.005 of S0's, its D and T within 0.4)

**Width, stratified:** 16 requests x j tokens at random steps, B = 16 j. It is an **optimistic** bound for B distinct
requests, because it includes the request affinity below.

| B | distinct experts D | tiles T | tiles per row T/B | vs B = 16 | fill |
|---|---|---|---|---|---|
| 16 | 57.07 | 57.07 | 3.567 | 1.000 | 0.140 |
| 32 | 71.96 | 72.70 | 2.272 | 0.637 | 0.220 |
| 64 | 83.15 | 88.73 | 1.386 | 0.389 | 0.361 |
| 128 | 91.24 | 117.57 | 0.919 | 0.258 | 0.544 |

**Verify blocks** (the 16 requests' k + 1 consecutive tokens), against the same requests and token count at random
steps. The interval is the spread of the block draws (their 2.5-97.5 %) over the reference cell's mean, not a confidence
interval on the ratio:

| k + 1 | rows | T | vs random steps (interval) | fill |
|---|---|---|---|---|
| 2 | 32 | 70.07 | 0.964 (0.903-1.008) | 0.228 |
| 4 | 64 | 85.46 | 0.963 (0.921-0.998) | 0.374 |
| 8 | 128 | 115.53 | 0.983 (0.968-1.000) | 0.554 |

**Request affinity:** the failed control's own row. A batch that lets requests contribute several tokens reads 0.9795 of
the stratified D at B = 16 (S1 0.9773). A request's tokens share experts about 2 % beyond separate requests'.

## What it says (reported, never ruling)

- **Rows per step are the lever.** Going from 16 to 64 rows cuts tiles per row to 0.39 of B = 16's, and to 128 cuts
  them to 0.26. Wider batches and speculative-decoding verify rows do this alike, because both put more rows on a
  sub-linearly growing expert set.
- **Consecutive-token sharing adds a few percent on top.** The mean ratios are 0.96-0.98. The intervals shown are the
  spread of single blocks, and no confidence interval on the ratio is computed, so no significance is claimed. Either
  way it is small next to the row count, and not a separate lever.
- **Request affinity is about 2 %.** Together with #1869's 0.3-0.6 % cross-request overlap, scheduling by expert overlap
  stays a small effect.
- **So a served lane worth registering** would read W32 or W64 decode, or a verify width, against W16 on the served
  step. The premise (cost tracks tiles) is what that read would test.

## Limits

- **The width bound:** this replay holds 16 requests, so B > 16 means 16 requests with several tokens each, not B
  distinct requests. Distinct-request widths need a wider replay.
- **Verify rows:** they are the recorded continuation (accepted tokens), not draft tokens.
- **Carried from #1869:** teacher-forced bf16 picks, one prompt set, one model.
- **The premise is inference.** No speed is measured here.

## Reproduce

```
python bench/tile-fill/tilefill.py --replay bench/p153/router-replay/rr.json.pt --out tilefill.json
```

It takes about 6 s and needs torch (to read the `.pt` file) and numpy. The output equals
[`tilefill.json`](tilefill.json).
