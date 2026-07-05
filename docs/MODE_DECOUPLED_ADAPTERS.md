# Mode-Decoupled Adapters: Train Here, Query There

This is a provenance-and-validation feature, not a serving framework.

## Summary

A user may need to train a QLoRA adapter under a low-memory storage/offload regime (say,
`nf4-offload` on a card the experts exceed) and then query or evaluate that same adapter under a
different regime — better fidelity, lower latency, different hardware. Whether the adapter
*transfers* across ExpertsNbit storage modes is an empirical question, and this repo does not
assume the answer. The train/query matrix scripts train one adapter per storage mode, record the
train mode in a provenance sidecar, evaluate every requested train/query pair, and mark every
storage-mode mismatch explicitly. Storage mode becomes part of adapter provenance; cross-mode
querying becomes an inspectable, warned, measured path rather than a hidden assumption.

What this is **not**: a claim that adapters are mode-independent, identical across modes, or
universally compatible. Same-mode query is the cleanest contract; cross-mode query is an
empirical path measured per host/model/dataset/run.

## Why this matters

ExpertsNbit makes the frozen base's storage scheme an explicit, validated parameter
(`quant_type`, checkpoint metadata, the README support matrix). That explicitness is what makes
this testable at all: "train where it fits, query where it answers best" can be measured instead
of assumed — and when it fails, the failure is attributable to a recorded mismatch instead of a
silent one.

## Terminology

- **train mode** — the storage scheme + offload setting the adapter was trained against
  (recorded in the sidecar).
- **query mode** — the storage scheme + offload setting the adapter is evaluated/served under.
- **storage-mode mismatch** — train and query storage schemes differ (`nf4`-trained, `int8`-
  queried). Allowed by the validation scripts, recorded on every result row, warned at load.
- **same-mode query** — train mode == query mode; the cleanest contract. For an offload-trained
  adapter, the same-storage *resident* query is its same-mode anchor (offload changes tensor
  location, not math).
- **upward transfer** — trained under a lower-fidelity scheme, queried under a higher-fidelity
  one (fidelity per the test-pinned reconstruction chain `fp4 < nf4 < fp8 < int8 < bf16 < fp16`).
- **downward transfer** — the reverse.
- **offload transfer** — same storage scheme, offload flag differs.

## Mode labels

`nf4`, `nf4-offload`, `fp4`, `fp4-offload`, `int8`, `int8-offload`, `bf16`, `fp16` — a canonical
storage scheme plus an optional `-offload` suffix. The parser
(`scripts/mode_matrix_common.py:parse_mode_label`) rejects anything else; ambiguous names are not
allowed.

## What this validates

- Adapter portability across ExpertsNbit storage/offload modes **under the stated conditions**
  (model, dataset, steps, seed, host — all recorded per row).
- That storage-mode mismatch is detected, recorded, and warned — including the unknown-provenance
  case (an adapter without a sidecar cannot be assumed same-mode).
- The train-memory vs query-fidelity trade: whether training under the cheap regime costs
  anything measurable when querying under a better one.

## What this does not validate

- Universal adapter compatibility, on any axis.
- End-task quality beyond the held-out validation split used here.
- Speed claims across hosts (decode timing is per-link; see the support matrix's host caveats).
- Production serving guarantees.
- Grouped-GEMM or non-ExpertsNbit loaders, or arbitrary PEFT adapters without sidecar metadata.

## Method

Three scripts, run in order (Option B split so a failed query leg never forces a retrain):

1. `scripts/train_mode_adapters.py` — one `experts4bit_qlora.train` run per train mode (the
   existing trainer, unmodified except a `SEED` env knob), each saving `adapter_best.pt` plus an
   `expertsnbit_adapter_metadata.json` sidecar: train storage mode, offload flag, model, dataset,
   split, seed, steps, LoRA config, host/GPU/torch/bitsandbytes/commit versions, training peak
   GPU, s/step, before/after eval, adapter hash, command line.
2. `scripts/eval_adapter_query_modes.py` — per query mode: one streaming load of the base in that
   mode, the base/no-adapter held-out eval (a fresh `ExpertsLoRA` is zero-delta at init), then
   each adapter applied in turn and evaluated. Mismatch between the sidecar's train mode and the
   query mode is computed per leg, warned, and written to the row — recorded, not blocked.
3. `scripts/summarize_train_query_matrix.py` — JSONL -> Markdown/CSV: the train x query eval
   table, delta-vs-baseline table, query-cost table, best-per-mode tables, and the
   same-mode/upward/downward/offload-transfer/symmetry observations.

Every result row carries the full schema in `scripts/mode_matrix_common.py`
(`REQUIRED_ROW_FIELDS` + eval/cost/env fields); failed and skipped legs are explicit rows with
reasons, never absent cells of unknown meaning.

## Matrix

First run: OLMoE-1B-7B, seed 0, host RTX A5000, 25/25 legs pass (raw:
`runs/results/portability/query_matrix.{jsonl,csv,md}`). Adapters trained under each row mode,
then queried under each column mode. **Single seed — every cell is an observation, not a
validated pair.** Provenance note: this first matrix ran via the direct chain that predates the
job runner, so its rows carry GPU/versions/host but not a commit field; the seeded phase-3
portability jobs (`docs/OLMOE_REPEAT_VALIDATION_PLAN.md`) run through the runner and carry full
provenance.

Held-out eval loss with adapter (lower = better; base-no-adapter ≈ 1.48–1.50 per query mode):

| train ↓ \ query → | nf4 | int8 | bf16 | fp16 | fp4 |
|---|---|---|---|---|---|
| nf4 | **1.0208** | 1.0217 | 1.0271 | 1.0265 | 1.0354 |
| nf4-offload | 1.0216 | 1.0242 | 1.0245 | 1.0245 | 1.0361 |
| int8 | 1.0237 | 1.0171 | 1.0179 | 1.0173 | 1.0378 |
| int8-offload | 1.0226 | **1.0126** | 1.0147 | 1.0153 | 1.0360 |
| fp4 | 1.0210 | 1.0301 | 1.0314 | 1.0315 | 1.0348 |

Query-mode peak GPU is set by the query mode alone (the base loads in that mode), independent of
which mode trained the adapter: nf4/fp4 ≈ 4.96 GB, int8 ≈ 8.18 GB, bf16/fp16 ≈ 14.22 GB.

## Observations

All observed in this single-seed run; none proven. The seeded phase-3 jobs (nf4/int8 query
columns) will test whether these hold.

- **fp4 as a *query* mode degrades every adapter** — the fp4 column is uniformly the worst
  (1.0348–1.0378), ~0.01–0.02 above the same adapter queried under nf4/int8/bf16/fp16. Querying
  under a coarser codebook than you can afford costs quality here. Classified `quality_shift`.
- **Upward transfer (train coarser, query finer) roughly preserves same-mode quality.** nf4→int8
  1.0217 vs nf4→nf4 1.0208 (+0.0009); nf4→bf16/fp16 within ~0.006. Training under the cheap
  regime and querying under a finer one did not, in this run, cost much.
- **Downward transfer (train finer, query coarser) degrades a little more.** int8→nf4 1.0237 vs
  int8→int8 1.0171 (+0.0066). A mild asymmetry: in this run, upward transfer preserved better
  than downward — the direction matters, which is exactly why the matrix records the pair rather
  than assuming symmetry.
- **The single strongest cell is int8-offload → int8 query (1.0126)**, and int8-offload
  transferred strongly to every non-fp4 query mode. This echoes the grid's int8-offload
  observation but is *one seed* — it does not rank offload modes and does not survive as a claim
  until the seeded repeats.
- **Best query mode per train mode is always same-or-finer, never fp4.** Every row's best column
  is its own storage mode or a finer one; fp4 is never anyone's best query mode (including fp4's
  own adapter, whose best query is nf4).

## Seeded phase-3 (nf4/int8 columns, 3 seeds — claim_usable)

The seed-0 matrix above is one run. Phase 3 re-ran the nf4/int8 train rows' adapters (the
3-seed repeat adapters) queried under nf4 and int8 resident — 24 jobs, all `claim_usable`
(commit-attested via `E4B_COMMIT`; the gate passes this time). Cell = mean ± std eval with
adapter, n=3 (raw: `runs/results/portability_seeded.md`, `runs/query_jobs/`):

| train ↓ \ query → | int8-resident | nf4-resident |
|---|---|---|
| nf4 | 1.0316 ± 0.0094 | 1.0319 ± 0.0107 |
| nf4-offload | 1.0325 ± 0.0123 | **1.0280 ± 0.0057** |
| int8 | 1.0323 ± 0.0085 | 1.0398 ± 0.0098 |
| int8-offload | **1.0260 ± 0.0079** | 1.0321 ± 0.0106 |

What the seeds confirm — and correct:

- **The seed-0 absolutes were optimistic.** int8-offload → int8 was 1.0126 at seed 0; the 3-seed
  mean is 1.0260. Same lesson as the fp4-decode candidate: single-run cells drift on repeat.
- **Validated (3 seeds): the downward-transfer penalty.** An int8-trained adapter queried under
  nf4 degrades (int8→nf4 1.0398 vs int8→int8 1.0323, +0.0075; int8-offload→nf4 +0.0061), while
  nf4-trained adapters are query-mode-agnostic (nf4→int8 1.0316 ≈ nf4→nf4 1.0319). The
  asymmetry seed 0 hinted at holds across seeds: train-finer/query-coarser costs a little;
  train-coarser/query-finer does not.
- **int8-offload adapters transfer well — but CONFOUNDED (audit).** Best or near-best in both
  query columns, yet the audit's G-decomposition shows the load-bearing *resident* comparison
  (int8→int8 vs nf4→int8) is a **tie** (+0.07 G) — train precision buys nothing at certified
  placement. The int8-offload advantage lives only in the offload-trained rows, which §3 of
  `docs/MEASUREMENT_AUDIT.md` shows are running a different experiment (bf16 control: 0.0108
  best-eval gap on byte-identical weights). Downgraded to candidate pending debt D3. The
  **downward-transfer penalty and nf4-query-agnosticism remain validated** — those are resident,
  certified comparisons.
- **Not re-tested here:** the fp4/bf16/fp16 query columns (phase 3 covered nf4/int8 only), so the
  seed-0 "fp4-query degrades every adapter" stays a single-seed observation.
- **Caveat: the cross-mode gaps are within a standard deviation.** These are directional
  tendencies observed across 3 seeds, not large separations; reported as such.

### Portability status of the tested pairs

| pair class | status | note |
|---|---|---|
| same-mode query (diagonal) | measured pass | the cleanest contract; runnable and useful this run |
| upward (train 4-bit → query 8/16-bit) | measured pass, quality preserved | not yet `validated` (one seed) |
| downward (train 8-bit → query 4-bit) | `quality_shift` (mild) | small consistent degradation this run |
| any → fp4 query | `quality_shift` | fp4 query degrades every adapter |
| not-yet-tested regimes (e.g. offload query, fp8/bf16/fp16 train rows in repeats) | `not_tested` | outside this run's scope |

Statuses use the shared taxonomy: `validated` / `quality_shift` / `broken` / `impractical` /
`not_tested` / `blocked`. Nothing here is `validated` yet — that requires the seeded repeats.

## Guidance

- **Same storage mode**: the normal path. No warning.
- **Different storage mode, validated pair** (appears in a published matrix): allowed, with the
  documented expectation from that matrix — which is an observation, not a warranty.
- **Different storage mode, unvalidated**: experimental. The eval script warns and records; a
  user doing this in their own serving path should treat the result as unmeasured.
- **Missing adapter metadata**: warned — storage provenance unknown, same-mode cannot be assumed.

No general user-facing cross-mode load helper is added in this pass (scope): the warning and
recording behavior lives in the validation scripts, and existing adapter loading
(`experts4bit_qlora.infer`, `load_state_dict`) is unchanged.

## Known limitations

- One model, one dataset, one seed per matrix run; transfer observed here is
  host/model/dataset/run specific.
- The held-out eval loss is the only quality signal; no end-task suites.
- Offload legs measure the same math at different tensor locations; their eval deltas vs
  resident training reflect GPU nondeterminism accumulated over training, not offload math.
- Decode throughput is not measured by the matrix scripts in this pass (`decode_tok_s` is
  reserved in the schema).
- fp8 is present in the mode vocabulary but not in the first matrix's train rows.

## Reproduction

```bash
python scripts/train_mode_adapters.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --train-modes nf4,nf4-offload,fp4,int8,int8-offload \
  --steps 150 --seed 0 \
  --adapter-root runs/olmoe_mode_adapters \
  --out runs/olmoe_mode_adapters/train_results.jsonl

python scripts/eval_adapter_query_modes.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --adapter-root runs/olmoe_mode_adapters \
  --query-modes nf4,fp4,int8,bf16,fp16 \
  --out runs/olmoe_mode_adapters/query_matrix.jsonl

python scripts/summarize_train_query_matrix.py \
  --input runs/olmoe_mode_adapters/query_matrix.jsonl \
  --out-md runs/olmoe_mode_adapters/query_matrix.md \
  --out-csv runs/olmoe_mode_adapters/query_matrix.csv
```

Both grid scripts print their planned legs before running, support `--dry-run`, `--resume`, and
`--fail-fast`, and never download anything not implied by `--model`.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T16:48:44Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `8cb935bc7af6b3104789ebb452223708993a13f2614d15f80957d578c84d22f3` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T14:52:04Z` `9b736b4fb5d7ccb1194ddfff51c87ee59ce8b4489ec74d1476595d1f20cb70a0`
  - `2026-07-05T12:47:30Z` `45db13c636a21677550245709d5ef170d21225f48494105a8a7e919ae2aa2be9`
  - `2026-07-05T08:44:38Z` `f19b86ea6551dd28d67ca7dcde37728ac8bddea6e44a521d7a98e87c464b8d44`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[*&@#~O@&=%$0@~:.]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|    .ooo+oo*.    |
|   =o .  =+o+    |
|o * .+ .. E.     |
|.= o .o= o       |
|+ . o * S .      |
| o   o O *       |
|      o =        |
|       oo..      |
|      .o .oo     |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info MODE_DECOUPLED_ADAPTERS.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify MODE_DECOUPLED_ADAPTERS.md.ots MODE_DECOUPLED_ADAPTERS.md` succeeds against the on-disk bytes.
- Anchor file: `MODE_DECOUPLED_ADAPTERS.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
