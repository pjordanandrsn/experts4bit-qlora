# PREREGISTERED AMENDMENT — ∅-LADDER RE-PIN AT n=1024

```
status:   prediction/plan amendment — commit BEFORE any n=1024 eval run (R5)
trigger:  S9 fired. G_int8 = +0.0094 ± 0.0076 at n=64 (|t| = 1.24); MDE(3σ) = 0.023
          exceeded |G|. The instrument was under-resolved; the phenomenon is undecided.
cites:    AGENT_HANDOFF_POST_AUDIT_20260705.md (S9), D1/D2 report <commit>,
          MEASUREMENT_AUDIT_olmoe-qlora-grid-20260705-1351.md, plan-routed-v3 R-rules
owner:    Jordan        executor: coding agent
```

## 1. Design

- **New eval set: n = 1024**, drawn from the SAME source distribution and protocol as
  the original pinned set, **DISJOINT from the original 64**. The 64 are reclassified
  as a hypothesis-generating pilot: used for SD-based power analysis only (signs and
  means from the pilot informed nothing below). Pooled pilot+confirmation analysis may
  appear only as clearly-labeled secondary.
- Eval-set hash committed with this file. Tokenization and eval protocol identical to
  the pilot; eval determinism already certified bitwise at n=64.
- **Passes:** 6 modes × {resident, offload} + 1 determinism repeat = 13 (~15 min each).
  Offload passes extend the D2 bitwise certificate to n=1024. If pod time is
  constrained, resident-only + repeat (7 passes) is acceptable — the serve-placement
  certificate is a code-path property, already bitwise across 384 example-evals.
- **REQUIRED telemetry per pass:** per-example losses; per-layer routed sets per
  example (on-device bincount). **PREFERRED (one flag):** router logits/margins —
  fragility-index raw material.

## 2. Power basis (measured pilot SDs only; pilot signs unused)

- sd(nf4−int8) = 0.0610 → SE(1024) = 0.0019; MDE(3σ) = 0.0057.
  Pilot point estimate 0.0094 → expected |t| ≈ 4.9 if it holds.
- Tight trio: sd(int8−bf16) = 0.0180 → SE(1024) = 0.00056; "flat above int8"
  resolvable to ±0.0017 at 3σ.
- Cross-cluster pairs (sd ≈ 0.05–0.09): resolvable to ≈ 0.005–0.008 at 3σ.

## 3. Inference plan (pre-committed)

- **PRIMARY:** paired per-example nf4 − int8, two-sided, threshold |t| ≥ 3.
  Single primary → no multiplicity penalty.
- **CO-PRIMARY robustness:** Wilcoxon signed-rank on the same contrast (heavy-tail
  guard; flip-dominated d_i is the standing prediction).
- **SECONDARY (exploratory):** full 15-pair table under Bonferroni (per-test |t| ≳ 3.0).
- **Tail report:** fraction of Σ|d_i| carried by the top 5% / 10% of |d_i| examples;
  d_i histograms per pair.
- **Mechanism probe:** per mode pair, Spearman corr(|d_i|, per-example routing
  disagreement). Committed predictions: Jaccard(int8, bf16) ≫ Jaccard(nf4, bf16);
  corr > 0 for cross-cluster pairs; corr ≈ 0 within the {int8, bf16, fp16} trio.

## 4. Branch table (pre-committed)

| outcome | claim | consequence |
|---|---|---|
| G_int8 ≥ 3σ (≥ ~0.0057) | frozen-base precision gap is real | precision program resumes; portability percentages re-denominated in the new G **with SE propagated** (the 97%-forfeit and 0.80 G figures gain error bars); coverage recomputed only if G_total also clears |
| G_int8 < 3σ | **bounded null: frozen nf4→int8 gap < 0.006 (3σ) on this task distribution** | screening becomes a RESULT, not a pause; headline = calibrated flat axis; portability matrix reinterpreted as placement/provenance findings |
| trio within ±0.0017 | "flat above int8" ships at high resolution | licenses stop-at-int8 / stop-at-16-bit on fidelity grounds, independent of the primary branch |
| fp8 − {nf4, fp4} replicates ≥ 3σ | fp8 advantage promoted from scramble to finding | regularizer hypothesis earns its dedicated probe run |
| top 10% of \|d_i\| carries > 50% of Σ\|d_i\| | precision is a subpopulation phenomenon | those examples seed the precision-sensitive probe set; stratified follow-up design |
| corr(\|d_i\|, routing disagreement) > 0 (cross-cluster) | co-adaptation is partly a routing story | routing-pinned serve experiment promoted in the queue |

## 5. What will not be claimed

- No coverage ratio unless BOTH G_int8 and G_total clear 3σ (silly-ratio rule).
- No per-pair claims from the exploratory table without Bonferroni survival.
- No pooling of the pilot into the primary analysis.
- No training-side (adapter) claims — this amendment covers the frozen base only.
- No re-run, re-draw, or enlargement of the confirmation set after first analysis.

## 6. Ledger note (adopted rule)

No significance forecasts without the instrument's n in hand. The prior forecasts
(75% G clears; 10–15% S9) missed on a knowable instrument parameter, not on physics.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

- **Committed eval set**: `train[10064:11088]` of `tatsu-lab/alpaca` under the pilot's exact
  protocol (SEQ=256, response-only labels, all-masked-example filter). Post-filter
  n = **1024 exactly** (the filter dropped nothing). Disjoint from the pilot's
  `train[10000:10064]` by construction.
- **Committed eval-set SHA-256** (computed by `scripts/eval_null_per_example.py
  --hash-only`, which hashes the exact tokenized `input_ids`+`labels` sequence the eval
  consumes; every job self-attests the same hash in `result.json`):

  ```
  3e836c1a01ab5cce90b7034477f174f5058f4cd4c1690dcc25b01741dc1a851f
  ```

- In-repo name mapping: `AGENT_HANDOFF_POST_AUDIT_20260705.md` = `docs/POST_AUDIT_WORK_QUEUE.md`;
  the D1/D2 report `<commit>` = 5e8f831 (`runs/results/postaudit/null_ladder_per_example.md`);
  `MEASUREMENT_AUDIT_…` = `docs/MEASUREMENT_AUDIT.md`. plan-routed-v3 remains not-in-repo
  (see POST_AUDIT_WORK_QUEUE.md filing notes).
- **Instrument**: `scripts/eval_null_per_example.py` (routed-set bincounts always on;
  `--router-telemetry` enables the PREFERRED margins — the manifest enables it).
  Manifest: `runs/job_manifest/null_ladder_1024_jobs.jsonl` (13 passes, full 6×2+repeat —
  pod time is not constrained). Analysis: `scripts/analyze_ladder1024.py` implements §3
  and prints the §4 branch table verbatim with each branch's outcome.
- **Host discipline**: all 13 passes on ONE pod, RTX A5000 (same architecture as the pilot
  and the original grid), per the T5(c) architecture confound.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T18:03:06Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `ca04ee4f6a2e53dbebd4f54d61cf68c4cc0db6a17fe32acdb520edbc2b23578a` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[&%.o??o$0%+?O~!@]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|              +  |
|             * = |
|    .       . O .|
|   . .       + = |
|    . . S . . =.+|
|   ..o o . o B.o.|
|   ..o= . . X +..|
|  o o=.  E * =.. |
|   =o.+.  o ++o  |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info NULL_LADDER_1024_AMENDMENT.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify NULL_LADDER_1024_AMENDMENT.md.ots NULL_LADDER_1024_AMENDMENT.md` succeeds against the on-disk bytes.
- Anchor file: `NULL_LADDER_1024_AMENDMENT.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
