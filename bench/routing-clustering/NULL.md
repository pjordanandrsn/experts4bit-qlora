# Cross-request routing clustering: the null, written down before any null draw

**Question** (reported only, never a ruling): at sixteen concurrent requests, do the requests decoded in the same step
share experts more than requests would without that alignment? Fewer distinct experts per layer-step means fewer
expert bytes read and fuller 16-row tiles. If the same-step alignment carries real clustering, scheduling that groups
similar requests could be a serving lever.

**Data:** `bench/p153/router-replay/rr.json.pt` (sha256 `3c59e72f`).
- `picks` is `[48 layers, 32 rows, 128 steps, top-8]`, int16 expert ids.
- Rows 0-15 are stream S0, P153's served tree arm, teacher-forced in bf16. Rows 16-31 are stream S1.
- The steps are the 128 decode steps P153's W16 slope counts.

S0 is the primary stream; S1 is a replicate, reported beside it.

**The statistic.** For one layer and one step, `D` is the number of distinct experts among the 16 rows' top-8 picks
(128 assignments, 128 experts).
- At 16 rows an expert can receive at most 16 rows, since a row picks an expert at most once. So K25's 16-row tiles
  number exactly `D`, and the tile count adds nothing at this width. The mean tile fill is `128 / D`.
- `D_obs` is the mean of `D` over the 48 x 128 layer-steps. It is already public in P153's README: 57.107 (S0) and
  57.052 (S1). What follows is new: the null.

**The nulls**, each 1,000 draws with seed 20261020, drawn per layer and per row independently:

- **A (primary): independent circular shift.** Each row's 128-step sequence (at each layer) is rotated by its own
  uniform random offset in 0..127.
  - This keeps each request's own routing and its step-to-step autocorrelation, since consecutive tokens of one
    request route alike.
  - It breaks only the alignment of the 16 rows on the same step. So A asks whether same-step rows share experts beyond
    what each request's own routing implies.
- **B (reported): independent permutation.** Each row's 128 steps are permuted at random. This also breaks the
  within-row autocorrelation, and is the weaker null.

For each draw, recompute `D` at every layer-step of the shifted (or permuted) picks, and take the mean over the
layer-steps.

**What is reported:**
- `D_obs / mean(D_null)`, overall and per layer, for A and for B, on S0 and on S1.
- The null's 2.5-97.5 % interval of `mean(D_null)` over the 1,000 draws, so the ratio carries the interval
  `D_obs / [q97.5, q2.5]`.
- A ratio below 1, with the interval clear of 1, means same-step rows cluster.
- The tile fill `128 / D`, observed and null.

**Bands, stated now, never ruling:**
- **Under 0.98, clear of 1:** clustering worth a scheduling look. That is at least 2 % fewer distinct experts.
- **0.98-1.00:** small.
- **Clear of 1 above:** anti-clustering, where same-step rows spread out more.
- **Interval including 1:** none detected at this resolution.

**Limits, stated now:**
- One prompt set (P109's 16 prompts), teacher-forced bf16 picks, one model (Qwen3-30B-A3B).
- All 16 rows sit at the same decode position in each step. So a position effect (routing that depends on the token's
  index) counts here as alignment. A rotates rows onto different positions, so A's ratio includes that effect; it is
  named, not separated.
