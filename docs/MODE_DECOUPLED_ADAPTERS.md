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

*To be filled from the first run's generated tables (see Reproduction). Until then this document
defines the method; it reports nothing.*

## Observations

*To be filled from the first run. Template discipline: "In this run, adapters trained under X
transferred to Y with [improvement/degradation] relative to same-mode query." "The result
suggests, but does not prove, that ..." This should be read as a storage-mode portability test,
not as a benchmark.*

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
