# RESULTS — T5 dispatch-algebra diet: REFUTED (revert)

Run 2026-08-24 against `PREREG-t5-dispatch-diet.md` (merged #218), on a
rented EPYC 9755 (48-core slice) + RTX 5090 (vast 48576023, $0.526/hr,
destroyed + verified zero). e4b at `7559f36`, gnf4 at `f08a66f`,
torch 2.13.0+cu130, transformers 5.15.1. Receipts in `receipts-t5/`.
Verdict computed by `t5_verdict.py` (self-tested, 10 branches).

## Verdict table

| bar | registered | measured | result |
|---|---|---|---|
| G0 A/A | < 7.5% | **1.49%** (219.0 / 215.8) | PASS |
| G1 token identity | bit-identical ×4 arms | identical | PASS |
| H1 attribution | ≥ 50% engine | **100%** (differential; see amendment) | PASS |
| H2 nonzero | ≤ 60/step | **0/step** (from 174) | pass |
| H2 churn cut | ≥ 15% | **−0.7%** (220.1k → 221.5k ops/step) | FAIL |
| H3 wall | ≥ 15% (partial ≥ 8%) | **3.63%** (215.34 → 207.53 ms) | FAIL |

**⇒ REFUTED (revert).** `dispatch_diet` stays default-off. The code
remains in-tree as a semantics-proven opt-in (G1 held), with no
performance claim attached.

## AMENDMENT-t5-h1 (registered before any H2/H3 number was read)

The stack instrument failed on the box: `with_stack=True` captured no
Python frames — all 2082 `aten::nonzero` rows grouped under
`<no-py-frame>`, so the registered frame-attribution H1 was
unmeasurable. Replacement: **differential attribution** — the two
profile arms differ by the engine flag alone, so every nonzero call the
B arm eliminates is an engine-dispatch call by construction. Same 50%
floor, strictly stronger identification. Measured: (174 − 0)/174 =
100%. The calculator keeps the stack basis whenever frames exist.

## What was learned (the negative result is the finding)

1. **Sync elimination is not a win on a busy-bound host.** All 174
   per-step dispatch syncs were removed, decoded tokens stayed
   bit-identical, and the wall moved 3.6%. A device→host sync only
   costs when the host would otherwise run ahead; this host has no
   run-ahead to lose — it is saturated doing Python/aten dispatch work.
   Stall-bound and busy-bound are different diseases; T5 treated the
   wrong one. (The T1-compile refutation said the same thing from the
   other side: guard tax, not launch tax.)
2. **The op spray does not live in the MoE dispatch.** The diet
   de-duplicated the MoE tier's gathers (index_select 743→577/step,
   arange 336→240/step) and the churn total did not move: `copy_`
   7,353/step and `.to` 6,218/step come from somewhere else. The
   per-bracket medians name it: **attention host = 124.6 ms of the
   215.8 ms step (57%) on this box** (43.8/139.9 = 31% on the t1b
   box) — the paged-attention shim's host path is the largest single
   bracket on both hosts, and it was never the T5 target.
3. **The amortization instrument tax is 7.4%** (231.7 ms amort-on vs
   215.8 off, same arm otherwise) — the first production-shape
   measurement in the campaign. Historical bench numbers carry ~7%
   instrument tax, not the 15–25% I predicted.
4. **Cross-box variance within "the same class" is large and now a
   named hazard.** This box's baseline is 215.8 ms where the t1b box
   read 139.9 (amort-on 231.7 vs 139.9): a 1.55–1.66× host-side gap on
   an allegedly same-class rental. The slice is CPU-weak: host triad
   204 GB/s, DRAM-bus bracket 33.5 ms vs t1b's 14.4. The A/A gate
   (1.49%) certifies internal validity only — **on-box A/B/A stands;
   cross-box absolute tok/s does not translate.** Consequence for the
   ladder: rent gates gain a host-side bar (triad ≥ 300 GB/s and a
   single-thread probe) before any box is called reference-grade, and
   SPEC-425's 37.6 ms target is only scoreable on a reference-grade
   host.
5. GPU side is healthy and measured: device triad 1570.8 GB/s on this
   5090 — the T3 kernel-headroom arithmetic (13.3 ms/step at ~400 GB/s
   achieved) keeps its ~3–4× ceiling.

## Prediction scored (made on the record before receipts)

Predicted: wall −25 to −35%, instrument tax 15–25%. Measured: −3.6%
and 7.4%. Both wrong, same direction (overestimated host-stall share).
The busy-bound/stall-bound distinction above is what the prediction
lacked.

## Ladder after T5 (SPEC-425)

- **T5b (next): the attention shim's HOST path.** 124.6 ms/step here,
  43.8 on t1b — largest bracket on both hosts. First stage is
  attribution with `torch.profiler.record_function` region markers
  (the stack instrument is proven broken on these boxes; region
  parents are robust), then registered edits against the named sites.
- **T3: NF4 grouped GEMV kernel** (gnf4): decode at B=16 runs the
  M-tile kernel with sm_86-tuned configs on sm_120; byte double-load;
  BLOCK_K sized for 100 KB SMEM on a 228 KB part. 13.3 → ~4–6 ms/step
  is the target arithmetic.
- **T4: CPU/GPU overlap** after the host is no longer busy-bound.
