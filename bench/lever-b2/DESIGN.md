# Lever B2: a multi-row K33 over device-grouped rows (design note, $0, no box)

**Status:** a design note, written before any code. It names the row grouping, the bitwise contract and its check, and
the kill criteria. Review: families (K33 and both families' shapes), then review-desk. No kernel, route or box moves
until it is reviewed. A kernel lane in grouped-nf4-gemm and a microbench read here follow, each registered on its own.

**Why now.** P227 (`bench/p227/PREREG-p227.md`, e4b#2036, read on `p227-5090-1`) timed the per-row route
`E4B_NF4_SMALLM_BW=1` (K33, one (token, slot) row per program) against the served tile route (K25) at 2 to 8 decode
tokens. Its B2 trigger, B's time over the byte floor at T = 8 above 1.25, fired on both families:
- Mixtral-8x7B: 2.00 (lower bound 1.97);
- Qwen3-30B-A3B: 1.59 (lower bound 1.57).

Mixtral fired with the larger margin, so it goes first.

## What the per-row route wastes

K33 serves one row per program. Rows that share an expert each stream its weights, so the route's weight bytes scale
with rows, not with distinct experts. From P227's pinned routing, the same tables its reducer pins
(`p227_reduce.ROUTE_SHA256`, `DISTINCT_TOTAL`, steps 0-59), the weight bytes a route reads, over the floor:

| family | T | rows / distinct | max rows on one expert | M = 1 (today's B) | M = 2 | M = 4 | M = 8 |
|---|---|---|---|---|---|---|---|
| Mixtral (8 layers) | 2 | 1.132 | 2 | 1.132 | 1.000 | 1.000 | 1.000 |
| | 4 | 1.455 | 4 | 1.455 | 1.073 | 1.000 | 1.000 |
| | 8 | 2.213 | 7 | 2.213 | 1.386 | 1.027 | 1.000 |
| Qwen3 (48 layers) | 2 | 1.061 | 2 | 1.061 | 1.000 | 1.000 | 1.000 |
| | 4 | 1.210 | 4 | 1.210 | 1.030 | 1.000 | 1.000 |
| | 8 | 1.620 | 8 | 1.620 | 1.179 | 1.027 | 1.000 |

A route that serves up to M rows of one expert per program reads sum_e ceil(rows_e / M) expert copies. That arithmetic
comes from the routing tables, not from a measurement. The distinct totals reproduce P227's pins exactly.

**What P227 measured beside it.** On Mixtral at T = 8, B ran at 2.00 x the floor while reading 2.213 x its bytes, so
some of its re-reads were cheaper than first reads (P227's own caveat: re-reads hitting L2). On Qwen3 the two were
nearly equal (1.59 against 1.620). So the table bounds B2's gain; it does not predict B2's time, which the microbench
reads.

## The kernel: K33 with an `M_ROWS` constexpr

One program serves (tile, n-tile, k-split), where a tile is up to `M_ROWS` rows of ONE expert:
- **The weights are loaded once per K-step and decoded once.** These are the `words` load, and the tree or prmt32
  decode of each of the 8 codes.
- **Each code is applied to all M rows before the next code is decoded.** For j = 0..7: decode code j, then for each
  row m in the tile, `part_m += code_j * x_m[j]`. So the M partial tensors stay live, never the 8 decoded codes. That is
  the register budget the plan sweep checks.
- **Each row keeps its own accumulator.** It is reduced exactly as M = 1 reduces it:
  `acc_m += tl.sum(tl.sum(part_m, axis=2) * am, axis=1)`.
- **Split-K** writes each row's fp32 partials in M = 1's layout, `ws[(k_split * T + row) * N + n]`, and the host reduces
  them in split order, as today.
- **PDL:** the launch takes `int4_b32._pdl_kw` keyed on the call's rows, as K33 does. B2 reads no PDL itself.

## The row grouping: the device tile table, with tiles of `M_ROWS`

The route reuses what K25 already consumes on the device-grouped path: grouped-nf4-gemm's `_tile_table_r1`, called
with `BM = M_ROWS`.
- **One launch builds it.** That launch writes the stable expert-major order (a row's rank within its expert is the
  count of earlier rows with that expert) and the (row0, rows, grp) tile slots, contiguous per expert. An expert with r
  rows gets ceil(r / M) tiles, and the last one is partly masked.
- **Activations are gathered through the order.** A tile's row i reads the activation of token `order[row0 + i] //
  top_k`, by index, not by copying rows. K33's `GATHER_DIV` arithmetic applies only in the unsorted singleton layout,
  so B2 takes the gather index instead.
- **Outputs land in sorted order**, as K25's do. The glue after the call (the act, down, and the combine through the
  order) is the device-grouped path's existing glue. So B2 slots in where K25 runs today, not where the singleton
  route runs.
- **`M_ROWS` is a launch constant per call:** the smallest power of two at or above the call's largest rows-per-expert,
  capped at a registered maximum (4 or 8, set by the plan sweep). Each decode bucket is a captured graph, so one
  compiled variant per bucket is the cost.

## The bitwise contract, and how it is checked

**The claim:** for every row, B2's output equals M = 1 K33's output for that row, bit for bit (`torch.equal`), at
every `M_ROWS`.

**Why it can hold.** Per row, B2 performs M = 1's floating-point operations, in M = 1's order, on tensors of M = 1's
shapes:
- the same K-step loop and masks;
- the same per-code multiply-accumulate into `part`, codes 0..7 in order;
- the same two-axis reduction, times absmax;
- the same bf16 store, or the same split-K partial layout and host reduction.

Only the weight load and the decode are shared across rows, and both are exact: integer loads, and codebook values
that are bitwise equal between the two decodes.

**Why it might not, and why it is tested rather than assumed:**
- Triton does not promise a reduction order across different programs. The shapes per row are kept identical so that
  the layout, and so the order, is the same.
- A compiler can fuse or reorder fp32 multiply-adds differently once M rows share a loop body.

**The check** is a GPU test in the kernel lane, before any timing, with no tolerance. For every row of a B2 call,
`torch.equal(B2[row], K33_M1[row])`, across:
- `M_ROWS` in {2, 4, 8};
- both decodes (tree, prmt32);
- both projections of Mixtral's and Qwen3's shapes (`_BW_SHAPES`);
- split-K on and off;
- eager and graph-captured;
- routing with partial last tiles, single-row experts and an expert with every row;
- P227's pinned T = 2, 4 and 8 cells.

M = 1 itself runs through the B2 kernel too, and must equal today's K33 bitwise. That proves the refactor changed
nothing for one request.

## Kill criteria (each decided by a registered read, never after looking at timings)

| # | criterion | consequence |
|---|---|---|
| KC1 | the bitwise check fails for any row at some `M_ROWS` | that `M_ROWS` is dead; if every M > 1 fails, B2 is dead (no tolerance is offered in its place) |
| KC2 | every K33 plan at `M_ROWS` = 2 spills registers (compiled `n_spills > 0`) on either Mixtral shape | B2 is dead on the 5090 at that M; a spilling plan is never selected |
| KC3 | at T = 8 on Mixtral, under P227's instrument and pinned routing, B2's time over the floor has its upper bound above 1.25 (P227's own trigger) | B2 does not close the gap it was opened for: dead on Mixtral |
| KC4 | at T = 2 or 4, B2/B's upper bound is above 1.00 | B2 must not lose where B already passes (Mixtral PASSed at both); that T stays on B |
| KC5 | any change to a default route | not allowed here: B2 is opt-in (a value of `E4B_NF4_SMALLM_BW`, or its own knob) until a served read licenses it |

## The order of work

1. **This note:** review (families, then review-desk), then the maintainer.
2. **The kernel lane** (grouped-nf4-gemm, its own PREREG):
   - the `M_ROWS` kernel and the tile-table call;
   - the bitwise tests above;
   - a plan sweep per M with the compiled register and spill counts.
   KC1 and KC2 are decided there.
3. **The microbench read** (here, its own PREREG). P227's instrument and pinned routing, with an arm B2 beside K and
   B, on one RTX 5090 at about USD 0.1 a box. It follows bench/common rules 1-7. KC3 and KC4 are decided there.
4. **A served read** only if 3 passes. B3 reads `E4B_NF4_SMALLM_BW=1` (B, not B2) on Mixtral at 2 and 4 requests,
   and is registered separately now. B2's served read would be a later lane, after B3.

## What this note does not decide

- **The plan per `M_ROWS`, and the cap of 4 or 8.** The kernel lane's sweep selects them.
- **Qwen3's B2.** Qwen3's B missed K25 at T = 2 and 4 in P227, and there is no Qwen3 B3. Qwen3 runs in the bitwise
  check and the microbench as the second family, reported, with no served lane opened from it.
- **The step outside the expert call.** That is the serving roadmap's census, as P227 says.
