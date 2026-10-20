# Tile fill and its amortization: the construction, written down before any computation

**Question** (reported only, never a ruling): at sixteen concurrent requests K25's 16-row tiles hold about 2.24 rows
(e4b#1869). How do wider batches and speculative-decoding verify rows change the count of tiles per useful row?

**The premise, labelled as inference (not measured here):** K25 decodes an expert's NF4 block once per non-empty
16-row tile, so a layer-step's expert cost tracks the tile count, while the useful work is the row assignments.
- At up to 16 rows per expert per step, the tile count equals `D`, the distinct experts.
- Above 16 rows per expert, a tile count needs `ceil(m_e / 16)`. This read reports both.

**Data:** `bench/p153/router-replay/rr.json.pt` (`3c59e72f`), stream S0 (rows 0-15, 128 steps, 48 layers, top-8),
teacher-forced bf16. S1 is reported beside it. One generator per stream, seeded at 20261020, 1,000 draws per cell.

**(a) Width.** For B in 16, 32, 64 and 128 rows, a batch is B (request, step) pairs drawn uniformly without
replacement from the stream's 16 x 128 pairs at one layer.
- #1869 found same-step alignment close to independent, so composing from different steps stands in for B independent
  requests.
- Reported per B, averaged over layers and draws, with the draws' 2.5-97.5 % interval:
  - `D(B)`, the distinct experts;
  - `T(B)`, the tiles: the sum over experts of `ceil(m_e / 16)`;
  - `T(B) / B`, tiles per row (the per-token decode work if the premise holds), with its ratio to B = 16;
  - the fill, `8B / (16 T(B))`.
- The B = 16 cell is checked against #1869's null A mean (57.26 for S0) as a positive control.

**(b) Verify blocks.** For k + 1 in 2, 4 and 8, a verify block is the sixteen requests' k + 1 consecutive tokens (steps
s .. s + k, with s uniform in 0 .. 128 - (k + 1)) as the rows of one step: 16(k + 1) rows.
- Reported, with intervals:
  - the block's `T` and `D`;
  - an independent composition of the same row count from (a);
  - the ratio, block over independent: below 1 means one request's consecutive tokens share experts beyond
    independence, so verify rows fill tiles faster than width alone.
- **A limit:** a served verify step's rows are draft tokens, which may differ from the accepted continuation. These are
  the recorded continuation, so this reads accepted-token sharing.

**Limits:** #1869's carry over (teacher-forced bf16 picks, one prompt set, one model), and the premise is inference.
This read informs whether a W32 or spec-verify served lane is worth registering. It measures no speed.

## Amendment 1 (before the amended computation): the width composition failed its positive control

The first run of (a) stopped at its positive control. The uniform composition of 16 (request, step) pairs read a mean
`D` of 55.90 on S0, against #1869's null A mean of 57.26. Drawing pairs uniformly lets one request contribute two or
three tokens to a batch, and a request's tokens share experts beyond chance even at non-consecutive steps: a
**request-level affinity**. So the composition does not stand in for independent requests, and (a) as constructed is
**VOID**. The failed control is kept and reported, never discarded.

**Amended (a'), stratified:** for B = 16j (j = 1, 2, 4, 8), each of the 16 requests contributes j tokens at distinct
uniform random steps, the same rows at every layer.
- The B = 16 cell (one token per request) must read within 0.2 of #1869's null A mean, or the script stops.
- Widths beyond 16 distinct requests cannot be read from this replay, which holds 16 requests. So B > 16 here is 16
  requests with j tokens each. It includes the request affinity, and is an **optimistic** bound on the amortization of
  B distinct requests.

**Amended (b):** each verify block (the 16 requests' k + 1 consecutive tokens) is compared with (a')'s same-row-count
cell: the same requests with the same number of tokens each, but at random steps.
- The ratio isolates **consecutive-token sharing beyond request affinity**.
- The uniform composition is reported beside it, as the failed control's own row.

**Also reported:** the request affinity itself, the uniform composition's `D` at B = 16 over (a')'s.
