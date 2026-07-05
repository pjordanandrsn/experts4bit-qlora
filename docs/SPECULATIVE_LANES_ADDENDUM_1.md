# SPECULATIVE LANES — ADDENDUM 1: FACTOR STRUCTURE (n=64 extraction)

```
status:   prediction/correction addendum — commit BEFORE the n=1024 telemetry joins
          are computed (R5). Tier: pre-collection if the 13 passes have not run;
          pre-analysis (state which) if they have. Do not edit the lanes plan or the
          re-pin amendment; this file supersedes the items named below by citation.
cites:    SPECULATIVE_LANES_PLAN_20260705.md <hash>
          PREREG_AMENDMENT_LADDER_REPIN_n1024.md <hash>
          n=64 D1/D2 report <hash>
```

## 1. Evidence (recovered from the n=64 paired summary statistics; verify on raw vectors — Z1)

Correlations of per-example deviations from bf16:

```
        fp4    nf4    int8   fp8    fp16
fp4    1.00   .744   .301   .584   .393
nf4           1.00   .319   .790   .522
int8                 1.00   .621   .743
fp8                         1.00   .719
fp16                               1.00     (Fisher SE ≈ .128 each at n=64)
```

Eigenvalues 3.32 / 0.99 / 0.35 / 0.23 / 0.11 (66% / 20% / rest). PC1 loads
near-uniformly on all formats; PC2 is bipolar {fp4, nf4} vs {int8, fp16} with fp8 at
0.006. One-factor triad on {nf4, fp4, fp8}: loadings 1.003 / 0.742 / 0.787, closing
to three decimals. Excess beyond one factor: ρ(int8, fp16) = .743 vs .165 predicted;
sd(int8−fp16) = 0.0132 is tighter than either format's distance to bf16.
Means: fp4 +.0223, nf4 +.0088, int8 −.0007, fp8 −.0094, fp16 −.0037.

## 2. Ledger grading (precise)

- **FALSIFIED (n=64):** the example-level independence gloss "the scattered formats
  flip different subsets." Deviations share a dominant common core; nf4 is a
  near-pure readout of it. Expert-level sharing (same experts vs same examples)
  remains OPEN until telemetry.
- **STILL OPEN, not falsified:** the committed prediction corr(|d_i|, flip count) ≈ 0
  within the {int8, bf16, fp16} trio. Under the two-factor reading this is expected
  to HOLD (factor 2 is the flip-free channel). The prior chat grading conflated
  deviation–deviation correlation with |d_i|–flip correlation; this entry corrects it.
- **STANDS:** the SD-clustering observation; the fp8-nearer-nf4-than-int8 expectation
  (now supported at the variance level).

## 3. Superseding predictions for the n=1024 telemetry (replace lanes-plan §S-A items)

- **P-A1** PC1 of the deviation matrix explains ≥ 50% of variance at n=1024.
- **P-A2** Spearman corr(PC1 score_i, total flip count_i across scattered formats) > 0.4.
- **P-A3** Two-channel claim, stated testably: ρ(int8, fp16) ≥ 0.5 replicates AND
  within-trio |d_i| remains uncorrelated with flip counts (factor 2 is flip-free).
- **P-A4** fp8 per-example flip count within ±25% of nf4's; mean-effect signs opposite
  (fp8 ≤ 0 ≤ nf4), conditional on the means resolving under the amendment's primary.
- **P-A5 (probe cross-validation)** The n=64 ranking by |loss_i(nf4) − loss_i(bf16)|
  — committed with this file (Z2) — predicts top-decile membership of the same
  quantity on the DISJOINT n=1024 set with ≥ 3× enrichment over chance (≥ 30% overlap
  vs 10% base rate).
- **P-A6 (conditional on Z3)** If compute paths are confirmed distinct, ρ(int8, fp16)
  stands as example-level smooth-sensitivity; if the bf16-native path branches
  differently from the dequant paths, a path-matched bf16 rerun collapses the
  factor-2 loadings.

## 4. Zero-GPU tasks

- **Z1** Rerun the factor decomposition on the raw n=64 vectors (on disk). Report
  Pearson vs Spearman side by side — the shared-outlier alternative to these
  correlations is the first thing to kill or confirm. Emit factor scores.
- **Z2** Emit and commit the probe-set candidate list (n=64, ranked |nf4 − bf16|
  deviation) BEFORE any n=1024 join. This is P-A5's timestamp.
- **Z3** Extend T4: source read of compute dtype / kernel branch per ladder mode.
  Specifically: does the native-bf16 forward share the matmul path of the
  dequant-based modes, or branch? One paragraph in OFFLOAD_MEMORY_FACTS or a sibling.
- **Z4 (interpretation gate)** The fp16 point's "activation-channel intercept" label
  in S-A is HELD until Z3 resolves.
- **Z5** Ledger file entry recording §2 verbatim.

## 5. What will not be claimed

- No factor-count claim ("two factors") beyond exploratory status at n=64.
- No mechanism claim for factor 2 until Z3 resolves.
- Nothing denominated in G; the amendment's primary contrast and branch table are
  untouched by this addendum.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

- **Tier: PRE-ANALYSIS, mid-collection.** The 13 n=1024 passes started at
  2026-07-05T18:08Z (first pass mid-flight at filing time); **zero telemetry joins,
  factor analyses, or per-example reads of any n=1024 output have been computed** —
  the only n=1024 artifacts touched were job-status lines and the eval-set hash
  attestation check. This addendum therefore precedes all n=1024 analysis, as R5
  requires, but not all collection.
- Cite hashes: `SPECULATIVE_LANES_PLAN_20260705.md` = `docs/SPECULATIVE_LANES_PLAN.md`
  @ ba23461; `PREREG_AMENDMENT_LADDER_REPIN_n1024.md` =
  `docs/NULL_LADDER_1024_AMENDMENT.md` @ 94c3931; n=64 D1/D2 report =
  `runs/results/postaudit/null_ladder_per_example.md` @ 5e8f831.
- Z-task outputs land in the follow-up commit (Z1 factor verification + scores, Z2
  probe-set list, Z3 paragraph in `docs/OFFLOAD_MEMORY_FACTS.md`, Z5 ledger at
  `docs/PREDICTION_LEDGER.md`), each citing this file.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T18:16:07Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `fbdc3e291d28e5058e496184535f532c07f40493af7805673dbc4b16962f4c30` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[$@!&~?+#:!+*?O.O]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|       +=. .EB+..|
|      o.....+***.|
|       o +.. Xo =|
|        o o . =+.|
|        So + oo..|
|        ..+ +  . |
|        .. o o   |
|         o..+    |
|          ooo.   |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info SPECULATIVE_LANES_ADDENDUM_1.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify SPECULATIVE_LANES_ADDENDUM_1.md.ots SPECULATIVE_LANES_ADDENDUM_1.md` succeeds against the on-disk bytes.
- Anchor file: `SPECULATIVE_LANES_ADDENDUM_1.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
