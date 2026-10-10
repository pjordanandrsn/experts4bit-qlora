# Training routing trace, Qwen3-30B-A3B on an RTX A2000: INCONCLUSIVE by the registered bar; the frozen ceiling is 1.09x the null

Plan: [`README.md`](README.md) (plan sha `11bef3f3`). Recorder and replay at e4b `450b07ff` (#1582, #1587, #1591,
#1609, #1618). Capture: the TC3 A2000 secondary arm `fused_attn4_m_offload_mb1` at `Qwen/Qwen3-30B-A3B @ ad44e777`,
micro-batch 1 x accumulation 8, `OFFLOAD_EXPERTS`, run 2026-10-10 20:18-21:21Z at `24140cd7`. Counts and bytes only; no
timings. Every number below is re-derivable on CPU from the trace (sha256
`8f17420214d2dd9df3cf7f44904b7cae24a849fc366b8e6cf16e4676c5e1dfdc`) with [`results-2026-10-10/commands.sh`](results-2026-10-10/commands.sh);
the outputs are in [`results-2026-10-10/`](results-2026-10-10/).

## What was scored

- **The capture ended early.** It hit an out-of-memory error in step 18 (index 17): the card carried ~1.5 GiB of other
  resident tenants, absent when TC3 ran this arm. The first 17 optimizer steps are complete, so those are scored
  (`replay.py --complete-steps 17`, #1618). The cut was decided before any hit rate was computed. It is set by
  micro-batch length (step 18 holds the longest), not by routing.
  - The kept steps pass the completeness guard: 13,056 records = 17 steps x 8 micro-batches x 2 passes x 48 layers.
  - All 6,528 forward / recompute pairs route identically.
- **One deviation from TC3's argv:** `TC1_FREE_OUTPUTS=1`. It is liveness only and does not change routing. It did not
  move the peak either: the out-of-memory error was byte-identical to the attempt without it.
- **The bar** (registered before the capture, 2026-10-10 17:58Z):
  - The cache holds 768 of the 6,144 (layer, expert) rows.
  - It runs over every pass through the end of step s, then is FROZEN.
  - Step s+1's forward touches are scored against the frozen set: one touch is one distinct row per micro-batch.
  - LFU, scored steps 6-17 (12 transitions; the plan's 6-20 shortened by the cut). The shuffled null (seed 1) is
    scored on the same window.

## Verdict: INCONCLUSIVE

| registered reading | threshold | observed |
|---|---|---|
| LFU hit rate / null's | BUILD-SIGNAL needs >= 2.0; NO-BUILD if <= 1.25 | 0.1920 / 0.1415 = **1.36** (73,662 and 54,264 of 383,571) |
| median per-layer step-to-step Jaccard | BUILD-SIGNAL needs >= 0.5 | **0.91** |
| Belady / null (the registered "ceiling") | NO-BUILD if <= 1.25 | 0.1776 / 0.1418 = **1.25** (1.2522) |

By the rule as registered, the verdict is INCONCLUSIVE. Its named next experiment is a 235B-slice trace.

## What the rule could not see (reporting; it does not change the verdict)

1. **Belady is not this metric's ceiling.** Belady is optimal for the live cache over the whole sequence, not for a
   set frozen at a step boundary, and here it scores below LFU (0.178 vs 0.192).
   - The true ceiling of the frozen score is each scored step's own top 768 forward rows
     (`oracle_frozen_reporting`; tested to bound every frozen policy).
   - It is **0.1922** on the trace and **0.1761** on the null: a ratio of **1.09**.
   - LFU already reaches the ceiling (0.1920).
   - So LFU's 1.36x over the null is mostly LFU doing badly on random routing (0.1415 against the null's own ceiling
     of 0.1761), not locality in the trace.
2. **The Jaccard clause does not discriminate at step granularity.** A step's per-layer forward set holds 112 of 128
   experts on average (8 micro-batches pooled). The null's median is 1.0, against the trace's 0.91.
3. **A training micro-batch touches most of a layer.**
   - It touches 82.9 of 128 experts per (micro-batch, layer) on average, with a range of 49-123 over 6,528 cells.
   - At 1/8 of the rows, no frozen set can serve more than 19.2% of the next step's forward.
   - For contrast, the #1469 decode traces (grouped-nf4-gemm `0677fddb`, 8 experts per token) reach LFU 0.45-0.66 and
     Belady 0.71-0.80 at the same 768 rows (code, dialogue, math, prose).
4. **Within-step reuse is real, but LFU does not keep it** (`replay.json`, live cache).
   - The dgrad touches each row its recompute just loaded. Belady serves it fully (1.0 at every budget).
   - LFU serves it at 0.026 (128 rows) and 0.188 (768), no better than its recompute. Freshly loaded rows have the lowest
     counts, so they evict each other before the dgrad arrives.
   - Whole-layer staging under `OFFLOAD_EXPERTS` already captures it. The recompute re-stages the layer, and nothing
     evicts it before that layer's backward re-dequantizes (`engines/offload.py:39-43`).
5. **Load skew** (compute and load balance, not transfer).
   - The 16 busiest experts of a (step, layer) take 42% of its routed tokens: median 0.42, p10-p90 0.35-0.48,
     against 0.125 if uniform.
   - That top-16 set is only moderately stable step to step: median Jaccard 0.45 (p10-p90 0.28-0.68).
   - A separate measurement on the same model agrees. The prefill expert-GEMM shape census (#846, on SC2's prompts)
     finds one expert in the top-8 of 50-94% of a layer's tokens. It measures token-level skew, which drives compute
     and tiling, not step-to-step reuse; this bar does not test it.
6. **Staged bytes** (`replay.json`, `staged_bytes_per_step`) price a HYPOTHETICAL row-granular tier of the cited
   layout, at 2,654,208 bytes per expert row. `OFFLOAD_EXPERTS` stages whole layer stacks.
   - At the observed touch rate, a tier that staged only the routed rows would move about 35% fewer bytes than
     whole-layer staging, before any residency (1 - 82.9/128).
   - That is a separate lever from caching. It is not measured here and carries no timing.

## Scope

A 30B proxy. Qwen3-235B-A22B has 94 layers x 128 experts with top-8; reading these rates onto it is an extrapolation.
The prefetch-overlap question is separate: a residency NO-BUILD would not argue against overlap, which hides transfer
whatever the reuse.
