# Results — P65: which per-expert ranking survives a calibration-domain shift, and does activation entropy earn a place in S-C's selector? (RTX 5090, 2026-09-24)

Pre-registration: [`P65-PREREG.md`](P65-PREREG.md) (#717), with Amendments 1 (#721) and 2 (#723), both dated before
any reading data. Record: [#710](https://github.com/pjordanandrsn/experts4bit-qlora/issues/710). Receipts:
[`receipts/`](receipts/). The selector the decision rule wrote is in
[`docs/SPECULATIVE_LANES_ADDENDUM_4.md`](../../docs/SPECULATIVE_LANES_ADDENDUM_4.md). The OpenTimestamps-anchored plan
documents are not edited.

The reducer ([`p65_reduce.py`](p65_reduce.py)) ran on the box. Re-run over the committed receipts (the three censuses
gzipped, the form the reducer reads), it reproduces [`receipts/p65_verdicts.json`](receipts/p65_verdicts.json) and
[`receipts/p65_table.md`](receipts/p65_table.md) byte for byte. Every number below is from those two files. The
reducer does not grade the predictions; the grading below is by hand against them, and every figure it uses is in
the table.

## Runs

| run | what | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|
| `p65-prove-1` | proof | host refusal rc 13: scale-and-add 0.152 s against 0.15 | $0.0419 | `a458e40` |
| `p65-prove-2` | proof | refused by the launcher before renting (wrong exclusion class; run id burned, adertha-agents#112) | $0 | `d249c87` |
| `p65-prove-3` | proof | host refusal rc 14: egress 97.8 MB/s against 100 | $0.0590 | `c3725bd` |
| `p65-prove-4` | proof under Amendment 1 | **PASS** rc 0 | $0.0577 | `8028a1f` |
| `p65-5090-1` | reading | host refusal rc 14: single-stream egress 21.8 MB/s | $0.0648 | `21a5dae` |
| `p65-prove-5` | proof at `ee3f006` (Amendment 2) | NOT_RUN: pre-flight ssh never authenticated | $0.0377 | `1fc2d55` |
| `p65-prove-6` | proof at `ee3f006` | **PASS** rc 0 | $0.0662 | `5300b78` |
| `p65-5090-2` | reading | host refusal rc 14: 4-stream egress 53.5 MB/s against 80 | $0.0253 | `9def15f` |
| `p65-5090-3` | reading | host refusal rc 14: 40.0 MB/s | $0.0352 | `0c0080e` |
| **`p65-5090-4`** | **the reading** | **rc 0: all three censuses complete and self-checked** | **$0.5491** | `33a7a0d` |

- **Lane total:** $0.9369 against the amended $2.05 ceiling. Proofs cost $0.2625 against their $0.45 budget. Every rented run has a proven teardown ([`receipts/teardown-proof.json`](receipts/teardown-proof.json) is the reading's).
- **The reading's box.** One RTX 5090 (575 W, driver 580.119.02) on an AMD EPYC 9655 with 105 GiB available RAM. The pre-flight read 4-stream egress at 129.2 MB/s and the host scale-and-add at 0.010 s. The Hessian budget was 42 GB.
- **Software.** e4b 0.37.3 at `ee3f006` (Amendment 2's merge, the same commit as the passing proof); grouped-nf4-gemm 0.33.0 at `5ca1897`; transformers 5.16.1, bitsandbytes 0.50.1.
- **Wall time.** 51 min from pre-flight to done.
  - Granite: census 84 s over 32 layers, 15,360 rows.
  - OLMoE-Instruct: census 122 s over 16 layers, 12,288 rows.
  - Mixtral: the fetch 8.8 min, the bake 3.7 min, and the census 1,781 s over 16 layers, 1,536 rows.
- **Selfchecks:** the moment-derived ρ matched ρ from the raw rows to 1.79e-08 (Granite), 1.06e-08 (OLMoE) and 3.80e-08 (Mixtral).

## The read

Within-layer Spearman, averaged over layers, with 95 % CIs from resampling layers. A signal **survives** iff
`r_cross_half ≥ 0.5` and `penalty ≤ 0.15`.

| family | signal | r_split | r_cross_half [95 % CI] | penalty | survives |
|---|---|---|---|---|---|
| Granite-3.1-3B | entropy | 0.968 | 0.882 [0.860, 0.902] | 0.086 | yes |
| | rel_act | 0.974 | 0.932 [0.916, 0.945] | 0.043 | yes |
| | freq | 0.984 | 0.891 [0.863, 0.914] | 0.093 | yes |
| | rho_x_err | 0.974 | 0.910 [0.892, 0.928] | 0.063 | yes |
| OLMoE-1B-7B-Instruct | entropy | 0.964 | 0.800 [0.764, 0.835] | **0.164** | **no** |
| | rel_act | 0.982 | 0.931 [0.921, 0.941] | 0.051 | yes |
| | freq | 0.935 | **0.380** [0.310, 0.453] | **0.556** | **no** |
| | rho_x_err | 0.980 | 0.928 [0.914, 0.940] | 0.052 | yes |
| Mixtral-8x7B (16 layers) | entropy | 0.927 | 0.789 [0.708, 0.853] | 0.138 | yes |
| | rel_act | 0.878 | 0.719 [0.651, 0.791] | **0.159** | **no** |
| | freq | 0.757 | **0.310** [0.215, 0.403] | **0.446** | **no** |
| | rho_x_err | 0.918 | 0.724 [0.653, 0.795] | **0.194** | **no** |

| family | entropy ~ rel_act (wiki / c4) | Colla-Q's claim: r_cross(entropy) − r_cross(freq) | RTN tail, c4val1 | premise | **selector** |
|---|---|---|---|---|---|
| Granite | −0.000 / −0.055 | UNRESOLVED, −0.009 [−0.044, +0.024] | 0.0142 | holds (P44-a P3 0.0992) | **TWO_ARMS** |
| OLMoE | 0.427 / 0.545 | **REPLICATES**, +0.421 [0.329, 0.501] | 0.2119 | no (this census) | NOT_WRITTEN |
| Mixtral | 0.326 / 0.332 | **REPLICATES**, +0.479 [0.369, 0.588] | 0.2344 | no (P44-a P3 0.2266) | NOT_WRITTEN |

**Colla-Q's cosine** was 0.9995 (Granite), 0.9997 (OLMoE) and 0.9999 (Mixtral) for entropy. Beside it, the
rank correlations above run 0.789–0.882. The descriptive signals (`rel_gu`, `rel_dn`, `rho_in`, `h_energy`) are in the
table file.

### Predictions

| | registered | read | verdict |
|---|---|---|---|
| P0 instrument | selfcheck + complete census, per family | all three complete; selfchecks ≤ 3.8e-08 | **HELD** |
| P1 `rel_act` survives on every family | r_cross ≥ 0.8, penalty ≤ 0.10 on Granite, OLMoE; ≥ 0.7 on Mixtral | Granite 0.932 / 0.043, OLMoE 0.931 / 0.051. **Mixtral 0.719 clears 0.7, but its penalty of 0.159 is past the registered survival line of 0.15, so rel_act does not survive there** | **HELD on Granite and OLMoE; REFUTED on Mixtral** |
| P2 entropy survives on Granite (≥ 0.8, ≤ 0.15); Mixtral ≥ 0.6; OLMoE no side | | Granite 0.882 / 0.086; Mixtral 0.789 (and survives, 0.138); OLMoE does not survive (0.164) | **HELD** (OLMoE: read, not predicted) |
| P3 entropy not `rel_act` again | \|ρ\| ≤ 0.8; ≤ 0.3 Granite; [0.3, 0.7] OLMoE | 0.000 / 0.055; 0.427 / 0.545; Mixtral 0.326 / 0.332 | **HELD** |
| P4 OLMoE: freq fails, claim replicates | | 0.380 / 0.556; +0.421 | **HELD** |
| P4 Granite: freq survives, claim unresolved | | 0.891 / 0.093; −0.009 [−0.044, 0.024] | **HELD** |
| P4 Mixtral: claim replicates | | +0.479 [0.369, 0.588] | **HELD** |
| P5 cosine ≥ 0.98 and r_cross below it | | 0.9995 / 0.9997 / 0.9999, against 0.79–0.88 | **HELD** |
| P6 Mixtral RTN tail on c4val1 ∈ [0.18, 0.27] | | 0.2344 | **HELD** |

- **What the refutation shows.** On Mixtral the error metric is the least stable of the families, even within one text: `r_split` is 0.878, against 0.974 and 0.982. The domain shift then costs it 0.159 more.
  - Its two parts each survive (`rel_gu` 0.855 / 0.103, `rel_dn` 0.750 / 0.113), but their root-sum-of-squares does not.
  - The rehearsal disclosed the same pattern on Granite-1B: each projection alone was more stable than the combination.
  - The refutation changes no selector, because Mixtral has no per-expert premise.
  - Had Mixtral had a premise, its table would put it in the **ENTROPY_ONLY** branch: entropy survives, is not redundant, and `rel_act` does not. That is the one family where the label-free signal transfers and the error metric does not.
- **P6's layers are verified, not inferred.** The registration found that the plan enumerates layers in checkpoint-key order, and it could only say P44-a's "16 of 32" Mixtral layers were most probably {0, 1, 2, 10–22}. P44-a's receipt (private store, `2026-09-19/p44-a-census/p44a/census_mixtral.json`) has rows for exactly those 16 layers, in that order, and so does this census. The two censuses read the same layers.

## Beside the rule (descriptive, gates nothing)

**Routing frequency redraws the residency engine's hot sets across texts on two of the three families.**
`hot_sets_from_profile` (the residency engine's own selection), fed a profile from one text, picks per-layer hot sets
whose Jaccard with the other text's is:
- Granite: 0.663 (top-4 of 40; chance ≈ 0.05);
- OLMoE: 0.141 (top-6 of 64; chance ≈ 0.05);
- Mixtral: 0.125 (top-1 of 8). **That is chance exactly: 1/8.**

A residency profile calibrated on one text therefore does not choose the hot experts of another on OLMoE or Mixtral.
Only Granite keeps its hot sets. This is a statement about rankings, not a measurement of residency hit rates or
speed, and no lane is registered on it here.

## What the decision rule did

- **Granite → TWO_ARMS.** Entropy, `rel_act` and `rho_x_err` all survive, and entropy is not redundant (|ρ| ≤ 0.055). The selector is written in [`docs/SPECULATIVE_LANES_ADDENDUM_4.md`](../../docs/SPECULATIVE_LANES_ADDENDUM_4.md): rank by `rel_act` and by `rho_x_err`, with routing frequency as the baseline arm, compared at matched bytes on an outcome. **Entropy earns a place, on Granite, as the ρ weight in `rho_x_err`.**
- **OLMoE and Mixtral → NOT_WRITTEN.** Neither has a per-expert premise: RTN tail fractions are 0.212 and 0.234, against P3's 0.10. Their read answers P4 and P5. Colla-Q's comparative claim, that entropy's ranking transfers across calibration text better than routing frequency's, **replicates on both**, including Colla-Q's own model.
- **Register rows:**
  - `e4b.quality.p65.granite.selector-two-arms.5090.2026-09-24`;
  - `e4b.quality.p65.collaq-stability.olmoe-mixtral.5090.2026-09-24`.
- **Unchanged.** Nothing here changes a default, a kernel or a stored format. N3's flip-attribution join is independent and stays open.

## What this read does not say

- **Ranking stability is not selector quality.** No mixed-precision cell ran, so nothing here says which arm recovers more quality at matched bytes. That is the addendum's cell to run.
- **The served rankings.** They are rankings of the NF4-served model (the registration's disclosed scope), not of the bf16 checkpoint.
- **Scope.** Two texts, 32 × 512 tokens per half. Mixtral covers 16 of 32 layers, with 8-expert ranks.
