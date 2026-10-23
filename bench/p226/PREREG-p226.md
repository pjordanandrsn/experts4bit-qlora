# P226: the K25 plan at bucket 32 — does any tile plan beat c1 at 256 routed rows on Qwen3-30B-A3B?

**Status:** registered before any data once merged. A microbench lane for the serving roadmap's item E, at 32 rows
(E32). A GO opens two later pieces of work, each its own PR: an opt-in per-bucket K25 plan table in e4b, and a served W32
read with its own registration. This lane changes no default, no gate and no speed claim, and it quotes no gain into the
served step. It needs one independent review, then the maintainer's pass, before any reading.

## Why

[P160](../p160/RESULTS-p160.md) read GO on C1, a K25 tile plan pair, at 16 token rows, and [P161](../p161/RESULTS-p161.md)
made it qwen3_moe's served default (`c1`: gate_up 64/256/4/2, down 128/128/8/3; block_n / kc / warps / stages). The plan is
per family, not per bucket: at bucket 32 the served decode sends 32 x top-8 = **256 routed rows** through K25, and it takes
c1 there too, a plan chosen at 128 routed rows.

**The question:** at 256 routed rows, on 32 independent requests' routing, does any plan in P156's 144-plan grid beat c1
by at least 5 %, under P160's instrument?

What this lane does NOT cover:
- **Bucket 64.** e4b takes K25 only up to 256 routed rows (`hot_residency._fused_over_stack`, `R_rows <= 256`). At bucket
  64 (512 rows) the served route is the M-tile GEMM (`nf4_mtile_captured`), confirmed by served route counters. That is a
  ROUTE question with a different accumulation (not bit-identical by contract), with its own USD 0 falsifier first.
- **Mixtral or other families**, other cards, other buckets. One RTX 5090, Qwen3-30B-A3B @ad44e777, bucket 32.

## The data: captured on the box, before any timing

P153's router-replay file cannot stand in for 32 requests. Its 32 rows are 16 prompts recorded twice, under two lm_head
variants: the same prompt shares 5.1 of its 8 picks across the halves, and the two halves together give 66.1 distinct
experts a layer-step, against 57.1 at 16 rows (a uniform draw of 32 rows would give 111.7). Every step in the file has
also been read once, by P156 and P160. A sweep on it would see too few distinct experts and too-full tiles, and so favour
large tiles.

So the reading box captures its own routing first (`p226_capture.py`, over [P154](../p154/PREREG-p154.md)'s routing pass,
staged byte-identical):
- **32 distinct requests:** `wikitext_rows(tok, n=48)[16:48]`, P109's builder (512 prompt tokens each, row k from token
  k x 4096 of wikitext-2-raw test). Rows 0-15 are P109's and P153's prompts; rows 16-47 have never been decoded by a lane.
- **The served default** at `E4B_PAGED_MAX_SEQS=32` (so a bucket-32 decode exists), graphs off (so a forward hook fires on
  every router), with `DEVICE_GROUPING` set as `serve_paged` sets it for captured decode. The int8 head (qwen3_moe's
  default) is recorded, not changed.
- **The window:** P152's bracketed form (`census_window`): ramp to a full 32-row decode, skip 32 steps, record 128. Each
  row's budget is 256 tokens (at B = 32 the first-admitted row is up to 31 tokens ahead; 160 would retire rows inside the
  window).
- **Asserted over the window, else the capture faults and the reducer VOIDs:** 48 routers hooked, 128 steps of 32 rows x
  top-8; 2 K25 calls a layer a step; the library's route record only `nf4_k25|le256`; its plan record only `c1|le256`; an
  NF4 store on the GPU baked on this box.
- **The record:** the picks as one `[48, 32, 128, 8]` int16 tensor, hashed (sha256 of its `<i2` bytes) into `capture.json`
  before the bench starts. The bench re-hashes the file it times, and the reducer requires the two to match.

Eager decode may choose tokens a graphed server would not (e4b's served tokens depend on the schedule). The bench needs
the routing distribution of 32 independent requests, not bit-exact served tokens.

## The instrument (`p226_bench.py`)

P156's sweep and P160's instrument, at 32 token rows. K25 (grouped-nf4-gemm v0.50.0, `nf4_smallm.gemm_nf4_grouped_smallm`,
decode `prmt`) is called exactly as the served route calls it on the NF4 store: gate_up reads the 32 token rows through
the lean tile table's `order` (`gather_div=8`); down reads the sorted rows and stores to the caller's order
(`scatter=order`); TF32 on fp32 weights. The tables are the served builder's (`build_group_tiles_fused(ids, 128, 16,
lean=True, sorted_ids=True)`). At 32 rows an expert can take more than 16 routed rows, so the tile check is one 16-row
tile per 16 rows of each expert (a fault VOIDs). Full-size random NF4 stores for all 48 MoE layers, so no replay is
served from L2.

1. **Selection (reported, never ruling).** Every plan of P156's 144-plan grid, per projection, in one seeded shuffled order
   (seed 226), on 8 selection steps (0, 8, ..., 56). Each plan's outputs are compared with c1's at step 0, bit for bit, and
   only identical plans are timed (a plan changes tiling and staging, never the values). c1 opens and closes each
   projection's sweep (the drift probe, reported).
2. **The carry rule (`p226_reduce.select`).** Each projection is ranked by its selection time (the mean over the steps of
   each step's median of 10 replays; ties by grid index), c1's own plans INCLUDED. The ranked lists pair in order (1st with
   1st, 2nd with 2nd, ...); the pair equal to c1 is skipped; the first three pairs are carried as K1, K2 and K3. The
   reducer re-derives the carried pairs from the recorded sweep and VOIDs a record whose carried pairs differ.
3. **Confirmation (the ruling).** Steps 64-127, never touched by selection, one paired rep each, in its Williams row over
   six arms:

| arm | plan (gate_up / down) | role |
|---|---|---|
| **C** | c1 | the served default: the control |
| C2 | c1, captured again | the self-pair (the same cubin) |
| **K1** | the first carried pair | **the registered question** |
| K2, K3 | the second and third carried pairs | reported |
| D | today's plan (32/64/4/3) both | reported: c1's own gain at 256 rows |

Each rep captures the spacer S (C's calls with their own outputs) and every arm's graph (both GEMMs of all 48 layers),
then replays the arms interleaved, 10 timed replays each, each one right after an untimed replay of S. An NVML thread
samples SM and memory clocks, power and the enforced limit, tagged with the arm replaying. After the replays, every arm's
outputs are compared with C's, bit for bit.

## The rule (`p226_reduce.py`, fixed before data)

- **VOID** on any structure fault: the capture missing, faulted or unhashed; the bench's picks not the capture's; the grid
  not swept once per projection; the carried pairs not the rule's; a confirmation step also a selection step; a tile
  fault; a rep's order not its Williams row; a replay missing; a spacer count wrong; C's second capture not bit-identical;
  unread clocks; any enforced power limit under 575 W; not an RTX 5090 (sm_120) on 48 layers.
- **VOID** when the self-pair C/C2's median is outside [0.99, 1.01].
- **NO_GO, a contract finding,** when any arm's outputs differ from C's on any rep: no timing is ruled on.
- **GO** when K1's median C/K1 minus 2 SE of the median (P155's SE: the larger of the normal-theory value and a seeded
  2,000-resample bootstrap) is at least **1.05**, both raw AND clock-adjusted (a faster-clocked arm's clock gain taken away
  at 1:1). Otherwise **NO_GO**, and c1 stays at 32 rows.
- **Reported, never ruling:** K2 and K3; D/C; the clock labels; the compiled registers and spills; the drift probe; and
  the **headroom**: C's time against the byte floor of the captured distinct experts (P154's method: the mean distinct
  experts a step, summed over the layers, x each projection's bytes per expert, 1,769,472 and 884,736, over P141's copy
  rate of 1,498.9 GB/s). The headroom also frames E64's USD 0 falsifier, on the same data.

## Pins and the box

- e4b at the launch commit; grouped-nf4-gemm **v0.50.0** (`41ef086e`), the release that carries K39's `prmt` decode. The
  tripwire asserts the version and that qwen3_moe resolves to decode `prmt` and plan `c1` (`_K25_PLANS["c1"]` equal to
  P160's C1).
- Qwen/Qwen3-30B-A3B @ad44e777 and its NF4 arena, baked on the box (P39's `k8_bake.py`).
- The card: an RTX 5090 (cc 12.0, VRAM >= 30 GB), power limit >= 575 W (rc 17 refuses before the install), host RAM >=
  60 GiB, disk >= 150 GB. K25's row-exact GPU tests run on the card before the fetch, as the premise.
- Staged by sha256 (`staged-p226.sha256`): the runner, capture, bench and reducer; P154's routing pass; P136's box,
  SC1b's window, P109's box, P39's bake and calibration; K25's row-exact GPU tests.

## Cost and the stop

- Needs: fetch 1,210 s and bake 550 s (the slowest measured, x 1.5), the capture 900 s, the bench 1,800 s (P160's 260 s
  at 16 rows, about 520 s at 32, x 3.5; P156's 462 s is inside it). The runner refuses a step that cannot fit before the
  deadline.
- **The pre-install fit** (the repository's guard-fit rule), before any install: FIT = install 450 + 1,210 + 550 + 900 +
  1,800 + the 600 s reserve = 5,510 s; FIT_NEED = 5,510 x 1.25 = 6,887 s, else rc 41 (a refusal before the install, not a
  strike). Static: (boot 300 s + 5,510 s) x 1.25 = 7,263 s <= the guard.
- Guard 2.1 h (7,560 s); worst 2.1 x USD 1.05 + 66 GB x USD 0.011 = USD 2.93, under the USD 3 the maintainer set.
- Run ids `p226-5090-1` to `-3`. The first GO or NO_GO ends the lane; two box-started VOIDs end it at STOP. No top-up and
  no step change after data.
- Rentals follow the standing rules: host pinned, USD 15 per box, `rent-watch` armed at create, teardown verified gone.

## What this lane cannot say

- Nothing at bucket 64 or wider (a route question), and nothing for other families, cards or buckets.
- No served speed: a microbench GO opens a served W32 read, with its own registration.
- No default changes.
