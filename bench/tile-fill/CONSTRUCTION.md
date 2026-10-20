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
