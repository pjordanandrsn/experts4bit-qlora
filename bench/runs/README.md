# bench/runs — GPU Rental Experiment Receipts

This directory contains immutable receipts for GPU rental experiments. Each run
produces a receipt at `<UTC date>/<run-id>/receipt.json`, validated against
[`docs/run-receipt-schema.json`](../../docs/run-receipt-schema.json).

## Directory Structure

```
bench/runs/
├── ledger.jsonl              # Append-only index: one line per receipt
├── 2026-09-06/               # UTC date of run start
│   └── exp-qwen3-tp3/        # run_id
│       ├── receipt.json      # The immutable receipt
│       └── ...               # Artifacts referenced in receipt.artifacts
└── 2026-09-07/
    └── exp-mixtral-fp8/
        ├── receipt.json
        └── ...
```

## Append-Only Policy

Receipts are **never edited after creation**. If a receipt contains an error or
needs correction:

1. Create a sibling errata file: `<run-id>/receipt-errata-YYYY-MM-DD.md`
2. Link the errata from the original receipt's `decision.link` if applicable
3. The ledger (`ledger.jsonl`) is also append-only: corrections are new lines

This ensures audit trails remain intact and reproducible.

## The Ledger

`ledger.jsonl` is a newline-delimited JSON file with one entry per receipt. Each
line is a JSON object containing at minimum:

```json
{"run_id": "exp-qwen3-tp3", "date_utc": "2026-09-06", "role": "CTO", "cost_usd": 12.45}
```

The check script (`scripts/check_run_ledger.py`) uses the ledger for fast
daily-sum queries without walking the entire receipt tree. The ledger must stay
in sync with the receipts: every receipt's `run_id` and `cost_usd.actual` must
appear in the ledger.

## Governance

All runs are governed by the policy in
[`docs/COMPUTE-GOVERNANCE.md`](../../docs/COMPUTE-GOVERNANCE.md):

- Approval thresholds (who can approve what cost)
- Per-role daily ceilings and global daily budget
- Allowed providers and GPU classes
- Teardown requirements

`scripts/check_run_ledger.py` enforces the policy on every CI run.

## Receipt Schema

See [`docs/run-receipt-schema.json`](../../docs/run-receipt-schema.json) for the
full schema. Required fields include:

- **Identity**: `run_id`, `requested_by` (role + agent)
- **Purpose**: `why`, `hypothesis`, `preregistration`
- **Approvals**: `approvals` array with role, agent, estimate, Slack permalink
- **Execution**: `provider`, `instance_id`, `gpu`, `image`, `commits`, `command`
- **Timing**: `start_utc`, `end_utc`
- **Cost**: `cost_usd` (estimated + actual)
- **Outcome**: `result` (status enum + summary ≤ 2000 chars)
- **Artifacts**: `artifacts` array with path, sha256, bytes
- **Teardown**: `teardown_proof` (method + evidence)
- **Decision**: `decision` (adopt/refute/void/pending + link)

Additional properties are allowed for run-specific metadata.

---

**Policy**: [`docs/COMPUTE-GOVERNANCE.md`](../../docs/COMPUTE-GOVERNANCE.md)
**Schema**: [`docs/run-receipt-schema.json`](../../docs/run-receipt-schema.json)
**Check script**: [`scripts/check_run_ledger.py`](../../scripts/check_run_ledger.py)
