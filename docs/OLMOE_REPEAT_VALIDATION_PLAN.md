# OLMoE repeat-validation plan

Repeat validation for the candidate findings from the OLMoE-1B-7B storage-mode validation grid
(`docs/OLMOE_EXPERTSNBIT_GRID.md`). Scientific rule: confirm same-model candidate findings with
same-model repeats — Qwen3-30B-A3B is tracked separately (`docs/QWEN3_30B_EXPERTSNBIT_GRID.md`)
as the gated larger-model target, and no Qwen3 repeats launch unless explicitly instructed (its
int8-resident legs need 48 GB-class cards).

Status labels used throughout: **Stable** (held across seed-matched repeats), **Candidate**
(observed, not yet repeated), **Host-specific** (real but tied to the measured host/link),
**Needs repeat**, **Not claimed**.

## Findings entering this plan

| finding | entering status |
|---|---|
| int8-offload posts the best training eval (best 1.0140 vs ~1.024–1.030 elsewhere) | Candidate — single run |
| fp4 resident decode faster than nf4 (12.59 vs 10.12 tok/s, single sample each) | Candidate — decode is noisy |
| offload collapses the storage-width memory difference (2.52 vs 2.72 GB offload; 5.28 vs 8.50 GB resident) | Candidate, expected Stable + Host-specific |
| resident training memory scales with storage width | Candidate, expected Stable + Host-specific |
| BEFORE-training eval tracks fidelity ordering (int8 1.4811 < nf4 1.4905 < fp4 1.5041) | Candidate, mechanism test-pinned separately |
| s/step and tok/s absolute values | Host-specific by construction — never generalized |

## Phase 1 — core training repeats (12 jobs)

`{nf4, int8} x {resident, offload} x seeds {1337, 2027, 3407}`, OLMoE-1B-7B, 150 steps, the
grid's hyperparameters, one adapter + provenance sidecar per job. Purpose: separate mode effects
from seed variance on the two findings that matter most (int8-offload eval strength; the offload
memory floor), with the effective seed recorded everywhere. `torch.manual_seed` seeds both CPU
and CUDA generators; Python `random`/NumPy are not load-bearing in this training path; bitwise
determinism is NOT claimed.

Job ids: `train_olmoe_{nf4,int8}_{resident,offload}_seed{1337,2027,3407}`.

fp4 x 3 seeds is optional and LATER, not first. fp8/bf16/fp16 repeats wait until the first-pass
grid legs (still filling) say they matter.

## Phase 2 — decode repeats (3 jobs, run now: cheap and the fp4 result is surprising)

`decode_olmoe_{nf4,fp4,int8}_resident_repeat5`: one discarded warmup + 5 measured 128-token
greedy decodes per mode, one model load each (`scripts/decode_repeat.py`), mean/std/min/max
recorded. fp4-faster-than-nf4 is not claimed unless the repeat means separate by more than one
standard deviation each.

## Phase 3 — focused portability queries (24 jobs, gated)

Only after Phase-1 adapters exist on disk (the manifest generator physically refuses to emit a
query job whose adapter is missing): each of the 12 repeat adapters queried under `nf4` and
`int8` resident. Purpose: upward transfer (nf4-trained → int8 query), downward transfer
(int8-trained → nf4 query), offload-trained → resident query, asymmetry between the two
directions, and whether int8-offload's eval strength survives cross-mode querying. This is NOT
the full Cartesian matrix — expansion to fp4/fp8/bf16/fp16 columns is gated on these results.

## Expansion rules (gated, in order)

- **fp4**: joins the portability matrix only if Phase-2 or first-pass data shows a reason.
- **fp8**: wait for the first-pass grid legs; limited repeat only if clean and interesting.
- **bf16/fp16**: likely more useful as query columns than training rows; include only where they
  fit and produce a meaningful comparison; skip with an explicit reason otherwise.
- **Qwen3-30B-A3B**: separate campaign, explicit instruction required.

## What counts as confirmed

A claim graduates Candidate → Stable only when it holds in every seed-matched comparison of the
repeat set (rules printed in the summarizer's claim table). Stable here still means *stable on
this host class for this model/dataset/step budget* — the no-overclaim list in the grid doc
applies to everything.

## Reproduction

```bash
python scripts/make_olmoe_repeat_manifest.py --phase 1 --phase 2 \
  --out runs/job_manifest/olmoe_repeat_jobs.jsonl
# on each worker pod (see docs/RUNPOD_DISTRIBUTED_VALIDATION.md):
python scripts/runpod_claim_and_run.py --manifest runs/job_manifest/olmoe_repeat_jobs.jsonl \
  --jobs-root /workspace/runs/jobs --locks-root /workspace/runs/locks --pod-id "$POD_ID"
# controller, any time:
python scripts/summarize_runpod_jobs.py --jobs-root /workspace/runs/jobs \
  --results-root /workspace/runs/results
# after phase 1 drains:
python scripts/make_olmoe_repeat_manifest.py --phase 3 --jobs-root /workspace/runs/jobs \
  --out runs/job_manifest/olmoe_query_jobs.jsonl
```

## Results (bundle olmoe-qlora-grid-20260705-1351)

Phases 1 (12 train repeats) and 2 (3 decode repeats) complete, 15/15 pass. Aggregates:
`runs/results/olmoe_repeat_*.{csv,jsonl}`, `runs/results/summary.md`. Provenance: gate report at
`runs/results/provenance_report.json` — all 15 class `debug_only` on the per-job commit check only
(git-archive worker trees; see caveat below); metrics/env/GPU/versions fully captured.

### Training repeats — best held-out eval per seed

| mode | seed 1337 | seed 2027 | seed 3407 | aggregate best (mean ± std) | peak GB | s/step |
|---|---|---|---|---|---|---|
| int8-offload | 1.0181 | 1.0339 | 1.0262 | **1.0261 ± 0.0079** | 2.72 | 17.0 |
| nf4-offload | 1.0214 | 1.0346 | 1.0315 | 1.0292 ± 0.0069 | 2.52 | 14.7 |
| int8-resident | 1.0213 | 1.0383 | 1.0343 | 1.0313 ± 0.0089 | 8.50 | 11.9 |
| nf4-resident | 1.0211 | 1.0441 | 1.0287 | 1.0313 ± 0.0117 | 5.28 | 12.0 |

### Decode repeats — resident, 5 samples + 1 discarded warmup

| mode | tok/s (mean ± std) [min, max] | peak GB |
|---|---|---|
| fp4 | 12.87 ± 0.20 [12.61, 13.10] | 4.72 |
| nf4 | 12.68 ± 0.22 [12.29, 12.86] | 4.72 |
| int8 | 11.63 ± 0.02 [11.61, 11.65] | 7.95 |

### Claim status (summarizer's printed rules)

- **int8-offload posts the best training eval — OLMoE-supported, host-specific.** Best-eval win in
  3/3 seeds vs every other repeated mode; aggregate mean is lowest of the four. Its ~2.72 GB peak
  is near the 4-bit floor, so this is a low-VRAM/high-fidelity *candidate regime for OLMoE* — not
  a Qwen3 claim, and bf16/fp16-offload (single-run) are excluded from the ranking.
- **offload collapses the storage-width memory gap — OLMoE-supported, host-specific.** Offload
  width-delta 0.20 GB vs resident 3.22 GB (ratio 0.06), 3/3 seeds.
- **resident memory scales with storage width — OLMoE-supported, host-specific.** 3/3 seeds.
- **BEFORE-training fidelity ordering (int8 < nf4) — OLMoE-supported, host-specific.** 3/3 pairs.
- **fp4 decode faster than nf4 — NOT supported on repeat.** The single-run signal (nf4 10.12
  tok/s) was a slow outlier; repeat-5 puts fp4 (12.87 ± 0.20) and nf4 (12.68 ± 0.22) within a
  standard deviation. Reported as tied. This is the repeat grid doing its job — a candidate that
  did not survive.

### Provenance caveat

Per-job commit was not self-reported (workers ran `git archive` trees with no `.git`), so the gate
classes all 15 `debug_only` on the commit check. Metrics, environment, GPU, and library versions
are fully captured; the executed training path is functionally identical across the bundle's
branch commits (only difference: a gated no-op profiler hook the repeat jobs never enabled). The
runner now records commit via `E4B_COMMIT` (`scripts/runpod_claim_and_run.py`) so subsequent runs
self-attest. Seed-reproduction rests on the captured metrics, not on commit attestation.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T13:53:11Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `d9c0ecd7ab94cff03795cc7e25d16b1213a5de8d0bcf872630fe4748df1b33a2` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T09:22:23Z` `ec3a43133ebb7ce900a5775f7fc999b5cda590884972a6b0c9a6fe1176abe01d`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[!#&.?&!=%@#o&$$.]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|              .. |
|       o     ..  |
|        +    ... |
|       . + .ooo.o|
|        S =.o+*++|
|         o +.B+%+|
|          = +.@oB|
|         . E o++.|
|          . =o ..|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info OLMOE_REPEAT_VALIDATION_PLAN.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify OLMOE_REPEAT_VALIDATION_PLAN.md.ots OLMOE_REPEAT_VALIDATION_PLAN.md` succeeds against the on-disk bytes.
- Anchor file: `OLMOE_REPEAT_VALIDATION_PLAN.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
