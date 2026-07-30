# PREREG — flagship 30B-class training matrix across synthetic industry datasets (pre-data)

Registered **before** any matrix cell runs. Scope: does the newly built fused
training path (`nf4_qlora.FusedGroupedNf4` + `enable_fast_train`) train a
30B-class MoE correctly and competitively across datasets from disparate
domains, and what does it cost in time, memory and energy.

## Fixture

- **Model:** Qwen3-30B-A3B (48 layers, 128 experts, top-8), NF4 via the
  streaming loader. It is the only 30B-class model cached on the IL-1 volume;
  a second would need a ~50 GB download into ~30 GB of free space, so this
  matrix is **single-model by constraint, not by choice** — stated here so no
  reader infers a cross-model claim.
- **Card:** a 24 GB Ada-class GPU (sm_89) in the datacenter that hosts the
  model cache. A 32 GB Blackwell card failed to provision on three consecutive
  attempts, and reaching sm_86 would have meant giving up the cached model this
  phase needs. **Single-architecture by availability, not by choice.**
- **Config, fixed for every cell:** `offload=True` + gradient checkpointing
  (`use_reentrant=False`), seq 512, r=8, α=16, AdamW lr 1e-4, batch 1.
  Offload is mandatory: measured 27.30 GB unoffloaded vs 9.13 GB offloaded on
  a 24 GB card.
- **Steps:** 200 per cell (Tier B). Enough to rank; not a convergence claim.

## Datasets — 5 synthetic, disparate domains

Generated locally from templates, seeded, shipped with the receipts so every
cell is reproducible: **legal** (contract clauses → obligations), **clinical**
(symptom notes → structured triage), **finance** (transactions → risk
rationale), **code** (buggy snippet → fix + explanation), **support**
(customer message → resolution steps). 1,200 train / 200 held-out each,
disjoint by construction.

Synthetic, not scraped: the point is *disparate structure* (list-heavy vs
prose vs code vs numeric), not domain realism. **No claim is made that these
proxy real industry data.**

## Registered outcomes

- **B1 — bit-exactness (HARD GATE).** SHA-256 of every frozen expert's packed
  bytes before and after training must be **identical, 100 %**. QLoRA trains
  adapters only; a single changed byte in the frozen 4-bit stack means the
  training path is corrupting weights it must not touch. **Any mismatch voids
  every performance number in this document.**
- **B2 — loss parity, fused vs reference.** For each dataset, train twice from
  the same seed: once through the reference path, once through
  `enable_fast_train`. Registered band: **|Δ final-train-loss| ≤ 0.05** and the
  step-wise loss curves' median absolute difference **≤ 0.05**. The fused path
  reorders expert summation (group-sorted vs ascending id), so exact equality
  is not expected; a difference larger than this band means it is not the same
  computation.
- **B3 — cost, reported for every cell, not gated:** s/step, tokens/s, peak
  VRAM, and **energy** as J/step from `nvidia-smi` power sampling at 200 ms
  across the timed window (with idle baseline subtracted).
- **B4 — "best result", defined pre-data.** Best = **lowest held-out eval loss
  on that dataset's own held-out split**, reported per dataset.
  **Cross-dataset loss values are NOT comparable** — different token
  distributions make absolute loss meaningless across domains, and any table
  that ranks datasets by raw loss is reporting tokenizer statistics, not
  training quality. The only cross-dataset quantity reported is **relative
  improvement** (eval loss at step 200 ÷ eval loss at step 0) and the B3 costs.

## Amendment 1 (pre-data) — the volume was resized, so the matrix is two-model

The single-model clause above was true when written and is now false: the
backing volume was enlarged (~177 GB free), which removes the constraint that
forced one model. The matrix becomes:

- **Models:** Qwen3-30B-A3B (cached) **and** a second 30B-class MoE downloaded
  into the resized volume. 2 models x 5 datasets x 2 arms (fused vs reference)
  = **20 cells**.
- Everything else is unchanged: same config, same 200 steps, same gates
  B1-B4, same $25 cap.

No band moved. No data existed when this was written -- no matrix cell had
run, and the datasets had only just been generated. The storage enlargement is
a standing infrastructure decision, not a per-run cost of this matrix.

**Second-model selection is fixed HERE, pre-data**, so it cannot be chosen
after seeing which one flatters the result: the first of
`google/gemma-4-26b-a4b`, `mistralai/Mixtral-8x7B-v0.1`, `Qwen/Qwen3-30B-A3B-Instruct-2507`
that (a) downloads cleanly, (b) loads through the streaming NF4 loader, and
(c) fits the offload path on 24 GB. If none qualifies, the matrix ships
single-model and says so.

## Datasets, as generated (pre-data)

| dataset | n_train / n_eval | mean chars/example | sha256 |
|---|---|---|---|
| clinical | 1200 / 200 | 244.6 | `76fb9036de80` |
| code | 1200 / 200 | 231.5 | `e0176e044fb1` |
| finance | 1200 / 200 | 213.9 | `e90914aaedfc` |
| legal | 1200 / 200 | 449.3 | `379d6e521c7f` |
| support | 1200 / 200 | 438.7 | `fbc68d228750` |

Hashes registered so a later reader can confirm the cells trained on exactly
these bytes. One generator defect was found and fixed *before* any data: the
`code` and `support` generators had too few unique combinations to produce
1400 disjoint rows and the dedup loop could not terminate. The builder now
refuses to ship a short dataset rather than hanging or silently truncating.

## Rules

Bands do not move after data. Every cell ships whether it flatters the fused
path or not. If B1 fails anywhere, the matrix publishes the failure and no
speed/energy claim. If a cell OOMs or the pod dies, the shortfall is stated
and the cell is reported missing rather than silently dropped.

**Spend:** capped at **$25** for this matrix, under the standing $35/job
ceiling, with a session-independent backstop pinned to the pod id. Evidence is
mirrored to the QNAP at `/share/ZFS2_DATA/gnf4-evidence/n9-flagship/`.

## Known limitations, stated before the run

1. Single model and single architecture (both by availability, above).
2. 200 steps is a ranking budget, not convergence. No claim about final
   model quality.
3. Synthetic data — structural diversity only.
4. Energy is board power from `nvidia-smi`, not wall power: it excludes CPU,
   PSU loss and cooling, and is comparable **between cells here**, not against
   any external figure.
