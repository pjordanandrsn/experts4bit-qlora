# Evidence bundle: olmoe-qlora-grid-20260705-1351

OLMoE-1B-7B ExpertsNbit validation, 2026-07-05. Controller-aggregated from job-local results
(atomic-claim multi-pod execution; see docs/RUNPOD_DISTRIBUTED_VALIDATION.md and
docs/provenance_contract.md).

## Contents
- `olmoe_repeat_training_all.{jsonl,csv}` — 12 train repeats ({nf4,int8}x{resident,offload}x3 seeds)
- `olmoe_repeat_decode_all.{jsonl,csv}` — 3 decode repeats (nf4/fp4/int8 resident, 5 samples each)
- `olmoe_portability_all.jsonl` — (empty here; portability matrix is in portability/)
- `portability/query_matrix.{jsonl,csv,md}` — 25-cell train/query matrix (seed 0)
- `qwen3/qwen3_{nf4,int8}_decode.txt` — Qwen3-30B scale-transfer probe (resident fits; offload blocked)
- `provenance_report.json` — gate classification of every repeat job
- `summary.md` — full summarizer output (tables + claim status)

## Provenance status
Repeat/decode jobs: metrics/env/GPU/versions captured; per-job commit NOT self-reported
(git-archive worker trees), so the gate classes them `debug_only` on the commit check. Runner
fixed (E4B_COMMIT) for subsequent runs. Seed-reproduction rests on captured metrics. See each
doc's "provenance caveat".

## Graduated findings (OLMoE-supported, host-specific — 3/3 seeds)
- int8-offload posts the best training eval (agg 1.0261 ± 0.0079).
- offload collapses the storage-width memory gap (ratio 0.06).
- resident memory scales with storage width.
- BEFORE-training fidelity ordering (int8 < nf4).

## Deflated on repeat
- fp4 decode faster than nf4: NOT supported — tied on repeat-5 (fp4 12.87±0.20, nf4 12.68±0.22).
