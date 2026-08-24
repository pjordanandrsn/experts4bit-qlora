# RESULTS — T5 dispatch-algebra diet: REFUTED (revert)

Run 2026-08-24 against `PREREG-t5-dispatch-diet.md` (merged #218), on a
rented EPYC 9755 (48-core slice) + RTX 5090 (vast 48576023, $0.526/hr,
destroyed + verified zero). e4b at `7559f36`, gnf4 at `f08a66f`,
torch 2.13.0+cu130, transformers 5.15.1. Receipts in `receipts-t5/`.
Verdict computed by `t5_verdict.py` (self-tested, 10 branches).

## ⚠️ Operating-point disclosure (found post-verdict, pre-merge)

The arms omitted `--kv-batched`, so every T5 arm ran the PER-SEQ KV
append path — the G9-kvappend cert's REFUTED-side shape — not the
certified 122 tok/s configuration. The receipts prove the point: this
cycle's baseline (step 215.8, attention_host 124.6) matches the kvapp
cert's per-seq A arm (204.7 / 112.6) and not its certified B arm
(131.3 / 40.2). Root cause: `Fp8PagedKV(batched_append=False)` is
still the tree default and the bench only enables it via the flag —
the certified win is opt-in, and every post-cert probe that forgot the
flag (this one) silently regressed to the slow path.

Everything below binds to the measured (per-seq) point. The verdict is
internally valid there (G0/G1 held); whether the diet moves the wall
at the certified point is UNTESTED — the transfer argument (host is
still busy-bound there: ~131 ms host vs ≤ ~32 ms device) is
directional, not a measurement. Follow-ups registered from this catch:
flip `batched_append` default to True citing its cert, and every
future prereg pins the FULL bench command line plus a baseline
shape-gate (abort the arms when the baseline brackets land outside the
expected band — this cycle's baseline was 64% over the certified step
and no gate noticed).

## Verdict table (per-seq operating point)

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

## What was learned

1. **Sync elimination is not a win on a busy-bound host.** All 174
   per-step dispatch syncs were removed, decoded tokens stayed
   bit-identical, and the wall moved 3.6%. A device→host sync only
   costs when the host would otherwise run ahead; this host has no
   run-ahead to lose — it is saturated doing Python/aten dispatch.
   Stall-bound and busy-bound are different diseases; T5 treated the
   wrong one. (The T1-compile refutation said the same thing from the
   other side: guard tax, not launch tax.)
2. **The op spray does not live in the MoE dispatch.** The diet
   de-duplicated the MoE tier's gathers (index_select 743→577/step,
   arange 336→240/step) and churn did not move: `copy_` 7,353/step and
   `.to` 6,218/step come from elsewhere. At THIS (per-seq) point the
   biggest bracket is the per-seq KV append inside attention_host
   (124.6 of 215.8 ms) — i.e. mostly the storm the kvappend cert
   already solved, reintroduced by the missing flag. At the CERTIFIED
   point the brackets read attention_host 40.2 (31%) and
   **other_submission 76.4 ms (58%)** — the composite of MoE hot-path
   submission, dense projections, norms and router — which was never
   decomposed and is now the named T5b subject.
3. **The amortization instrument tax is 7.4%** (231.7 amort-on vs
   215.8 off, per-seq point) — the campaign's first production-shape
   tax number. Historical bench figures carry roughly this overhead.
4. **The "1.55× weaker box" read was wrong — it was the config.**
   Corrected against the kvapp receipts: same-config (per-seq)
   baselines are 204.7 (kvapp box) vs 215.8 ms (this box) — a 5.4%
   box-to-box gap, ordinary rental variance. The as-arm's DRAM bracket
   (33.5 ms) likewise matches the kvapp per-seq-era bracket (32.5),
   not a slow CPU. The real cross-box caution survives at ~5–8%
   scale, and the rent pre-gate keeps its host-triad bar, but no
   silicon was at fault here.
5. GPU side healthy and measured: device triad 1570.8 GB/s on this
   5090 — T3's kernel-headroom arithmetic (13.3 ms/step at ~400 GB/s
   achieved) keeps its ~3–4× ceiling.

## Prediction scored (made on the record before receipts)

Predicted: wall −25 to −35%, instrument tax 15–25%. Measured: −3.6%
and 7.4% — and the arms measured a different operating point than the
prereg named (missing `--kv-batched`). Three errors: overestimated the
host-stall share, overestimated the instrument tax, and failed to pin
the command line. The busy-bound/stall-bound distinction and the
shape-gate discipline are what those errors bought.

## Ladder after T5 (SPEC-425)

- **T5b (next): decompose and diet the host bill at the CERTIFIED
  point** — re-bracket with `--kv-batched` on, `record_function`
  region markers for attribution (profiler stacks are proven broken on
  these boxes), then registered edits against `other_submission`
  (76.4 ms) and the residual attention host (40.2 ms).
- **Default-flip PR: `batched_append=True`** — certified bit-identical
  by the kvappend cert; leaving it opt-in already cost this cycle its
  operating point.
- **T3: NF4 grouped GEMV kernel** (gnf4): decode at B=16 runs the
  M-tile kernel with sm_86-tuned configs on sm_120; byte double-load;
  BLOCK_K sized for 100 KB SMEM on a 228 KB part. 13.3 → ~4–6 ms/step
  is the target arithmetic.
- **T4: CPU/GPU overlap** once the host is no longer busy-bound.
