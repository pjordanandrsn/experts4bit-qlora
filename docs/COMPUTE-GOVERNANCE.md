# Compute Governance

This document defines the compute-spend policy for experiments and benchmarks
run by the AI agents in the `#ml-packages` workspace. All policy numbers are
provisional (`default-pending-owner`) until confirmed by the repository owner.

## Purpose

Expensive GPU-rental experiments require explicit approval before launch. This
policy ensures that:

- No agent can silently spend beyond their ceiling without oversight
- Each run has documented justification and hypothesis
- Approvals are recorded immutably in receipts
- Cost tracking is auditable across the entire ledger
- Teardown happens automatically to avoid forgotten instances

## Policy

All numbers below carry `status: "default-pending-owner"` in the machine-readable
policy ([`compute-policy.json`](compute-policy.json)) until Jordan confirms them.

### Approval Thresholds

Every run estimate requires approval **before launch**, recorded in the run's
Slack thread. The approver posts `APPROVE <run-id> $<est>` before the requesting
agent starts the instance. The receipt's `approvals` field records who approved,
their estimate, and the Slack permalink.

| Estimated cost (USD) | Approver required |
|----------------------|-------------------|
| ≤ $2                 | Requesting agent (self-approval) |
| ≤ $20                | One of: CTO, CSO |
| ≤ $50                | Two of: CEO, CTO, CSO |
| > $50                | Jordan (Owner) |

A run whose estimate exceeds the threshold without the required approvals cannot
launch. A run that launches with valid approvals but exceeds its estimate is not
retrospectively invalid — the approvals satisfied the policy at launch time.

### Hard Caps

- **Per-run hard cap**: $35 USD. No single run may be approved above this ceiling.
- **Per-role daily ceiling**:
  - CEO: $50 USD/day
  - CTO: $50 USD/day
  - CSO: $20 USD/day
  - CDO: $20 USD/day
  - COO: $10 USD/day
  - Scout: $0 (read-only, no experiments)
  - Warden: $0 (read-only, no experiments)
- **Global daily budget**: $100 USD/day across all agents and all runs.

A role's daily ceiling is the sum of `cost_usd.actual` from that role's receipts
dated on the same UTC day. The global daily budget is the sum of all receipts'
`cost_usd.actual` on the same UTC day. `scripts/check_run_ledger.py` enforces
these limits: a receipt that would push the role or global sum over the ceiling
fails the check.

### Allowed Providers and Hardware

- **Providers**: Vast.ai (verified + secure hosts only), RunPod (secure tier).
  Machine-readable: `["vast:verified-secure", "runpod:secure"]`.
- **GPU classes**: RTX 5090, RTX 4090, L40S, A100-80GB, H100.

Other providers or instance classes require explicit owner approval per run.

### Teardown Policy

Every rented instance **must** tear down automatically. Default wallclock limit:
2 hours; maximum: 6 hours. Teardown triggers:

- Completion (command exits)
- Wallclock timeout (the run's declared limit, ≤ 6 hours)
- Heartbeat loss (the instance stops reporting)

**Proof required**: each receipt's `teardown_proof` field must record the method
(e.g., `"vast-cli-destroy"`, `"runpod-api-terminate"`) and evidence (command
output, API response, timestamp). A receipt without teardown proof fails
`check_run_ledger.py`.

## Receipts and the Ledger

Every experiment produces an immutable receipt at
`bench/runs/<UTC date>/<run-id>/receipt.json`, validated against
[`run-receipt-schema.json`](run-receipt-schema.json). Receipts are also ingested
by the organisational corpus ([`pjordanandrsn/org-corpus`](https://github.com/pjordanandrsn/org-corpus)).
A result without provenance (missing receipt, uncommitted, or not in the corpus)
is observational only. Required fields include:

- `run_id`: unique identifier (referenced in the ledger and Slack threads)
- `requested_by`: `{role, agent}` — who launched it
- `why`: one-sentence purpose
- `hypothesis`: testable claim or question
- `preregistration`: URL or path to the pre-registration document
- `approvals`: array of `{role, agent, usd_estimate, slack_permalink}`
- `provider`, `instance_id`, `gpu`, `image`: rental details
- `commits`: `{repo: sha}` map of code versions used
- `command`: the exact command executed
- `start_utc`, `end_utc`: ISO 8601 timestamps
- `cost_usd`: `{estimated, actual}` in USD
- `result`: free text (≤ 2000 chars) + `status` enum (`OK`, `REFUSED`, `OOM`,
  `INSTALL_FAILED`, `LOAD_FAULT`, `HARNESS_ERROR`, `ALARM`, `NOT_RUN`)
- `artifacts`: array of `{path, sha256, bytes}` for outputs
- `teardown_proof`: `{method, evidence}`
- `decision`: `merge|continue|revise|abandon|rerun <url>` — a launcher-written receipt before any reviewer has decided uses `pending <work-id or url>`

Receipts are **append-only**. Corrections go in sibling errata files, never by
editing the original receipt.

The **ledger** (`bench/runs/ledger.jsonl`) is a single-line-per-receipt index:
each line is a JSON object with at minimum `{run_id, date_utc, role, cost_usd}`.
The check script uses it for fast daily-sum queries without walking the entire
receipt tree.

## Check Script

`scripts/check_run_ledger.py` enforces this policy:

1. Every `receipt.json` under `bench/runs/` validates against the schema.
2. Each receipt's `run_id` and `cost_usd.actual` appear in `ledger.jsonl`.
3. Per-role daily sums ≤ role ceilings; global daily sums ≤ global budget.
4. Each receipt's `approvals` satisfy the threshold for its `cost_usd.estimated`.
5. Each receipt has a non-empty `teardown_proof`.

Exit code 1 with `file:line`-style messages on any violation; `OK: N receipts, M days`
on success. The check runs in CI on every pull request and push to `main`.

## Out of Scope

This policy defines **what** can be approved and **who** approves it. The
**launcher** (CTO, separate issue) will enforce the policy at rent time,
implement the teardown guard, and emit receipts automatically. Until that
tooling exists, agents manually create receipts and record approvals in Slack.

---

**Status**: Policy numbers provisional (`default-pending-owner`) until owner confirmation.
**Machine-readable policy**: [`compute-policy.json`](compute-policy.json)
**Receipt schema**: [`run-receipt-schema.json`](run-receipt-schema.json)
**Ledger location**: `bench/runs/ledger.jsonl`
**Check script**: `scripts/check_run_ledger.py`
