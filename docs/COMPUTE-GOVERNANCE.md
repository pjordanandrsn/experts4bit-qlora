# Compute Governance

This document defines the compute-spend policy for experiments and benchmarks
run by the AI agents in the `#ml-packages` workspace. All policy numbers are
confirmed by Jordan (owner) on 2026-09-06T04:55Z.

## Purpose

Expensive GPU-rental experiments require explicit approval before launch. This
policy ensures that:

- No agent can silently spend beyond their ceiling without oversight
- Each run has documented justification and hypothesis
- Approvals are recorded immutably in receipts
- Cost tracking is auditable across the entire ledger
- Teardown happens automatically to avoid forgotten instances

## Policy

All numbers below carry `status: "confirmed"` in the machine-readable
policy ([`compute-policy.json`](compute-policy.json)), confirmed by Jordan on 2026-09-06T04:55Z.

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
  - CXO: $20 USD/day — a separate ceiling on top of the shared global budget (owner, 2026-09-06: "Cxo seperate budget on top of the shared budget is $20/day")
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

## How to launch

Agents rent through one entry point so the policy is checked *before* an
instance exists:

```bash
python -m experts4bit_qlora.tools.rent \
  --role CTO --agent cursor-desktop-mini/grok \
  --work-id experts4bit-qlora#433 --preregistration bench/p41/P41-PREREG.md \
  --hypothesis "…" --expected-result "…" --success-criteria "…" --failure-criteria "…" \
  --gpu "RTX 5090" --usd-per-hour 0.40 --wallclock-h 2 \
  --seat-executor CTO=cursor-desktop-mini/grok \
  --approval 'CTO/cursor-desktop-mini=https://cerin-amroth.slack.com/archives/C0BV5028SGM/p<16 digits>' \
  --approval 'CSO/ChatGPT=https://cerin-amroth.slack.com/archives/C0BV5028SGM/p<16 digits>' \
  --command "python bench/…" --dry-run
# live (Vast verified-secure / RunPod secure) is refused -- with a REFUSED receipt -- until
# E4B_RENT_LIVE arms the API adapter; --dry-run uses a fake provider and still writes a receipt.
```

`scripts/rent_run.py` is the same command. The launcher:

1. Estimates `$ / h × wallclock` and refuses over-ceiling, over global daily
   budget, a role without a numeric ceiling row, disallowed GPU/provider,
   missing approvals, or a cap above the $35 per-run hard cap without
   Jordan's approval. Every refusal writes a `REFUSED` receipt and a ledger
   line.
2. `--approval ROLE/AGENT=<slack-permalink>` is repeated as the spec
   demands: self-approval ≤ $2; one of CTO/CSO ≤ $20; two of CEO/CTO/CSO
   ≤ $50; Jordan above (Jordan also covers the hard cap, as role *and* agent
   `Jordan`). `approval_overrides` in `docs/compute-policy.json` tighten the
   spec while a named executor holds a seat -- today, while
   `CTO=cursor-desktop-mini/grok`, $2–$20 needs `all-of:CTO,CSO`; the
   launcher learns the seats from `--seat-executor ROLE=EXECUTOR` and records
   them in the receipt's `environment.seat_executors`, and
   `scripts/check_run_ledger.py` applies the same override to the receipt.
   **An undeclared seat is never the loophole:** the override applies unless
   every seat it names is declared and held by someone else -- omitting
   `--seat-executor` gets the stricter spec, in the launcher and in the check.
   Permalinks must be `https://cerin-amroth.slack.com/archives/C…/p<16
   digits>`; the launcher cannot read Slack, so the receipt carries them for
   the ledger check to resolve against the org-corpus raw layer.
   For a live run, the launcher takes the account lock before loading the
   budget ledger. Ledger readers and appenders use matching file locks. Every
   run, including a dry run, rejects an id already present in the ledger and
   atomically reserves its canonical receipt directory before provider create.
   Refused contenders receive unique attempt ids and cannot overwrite it.
3. Arms a teardown **guard on the controller** (`start_new_session`, not on
   the rented box), waits for the guard's arm marker (`guard-armed.json`;
   `--guard-arm-timeout-s`, default 60 s -- no marker means the launcher tears
   down WITHOUT running the command: `status HARNESS_ERROR`, `result invalid`,
   `reason guard-not-armed`; a process-start failure is recorded as
   `reason guard-spawn-failed` after synchronous teardown) and refreshes the guard's heartbeat every `timeout / 3`
   seconds for as long as `--command` runs. The guard destroys the instance
   on wallclock or heartbeat loss and proves teardown by listing the provider
   without that instance id; when the guard fires, the receipt says `status
   ALARM`, `result invalid` -- never `pass`. On normal completion the
   launcher tears down, writes the proof (`reason: completion`) and the guard
   exits. `pass` is written only when the launcher itself destroyed a live
   instance; an instance found already gone -- by the guard or by anyone else --
   is `ALARM` / `invalid`.
   Arm, firing and teardown files are accepted only when they name the current
   instance. A teardown proof is final only when it says complete and a fresh
   authenticated provider listing also proves that instance absent. The receipt
   schema defines an optional nested instance id for compatibility with historical
   receipts. New launcher receipts always include it, and both the launcher and
   canonical ledger checker validate that it equals the receipt's top-level
   `instance_id` whenever present.
   If synchronous teardown is not proven, the controller retains the account
   lock and retries destroy until an authenticated listing proves the instance
   absent; it never hands the lock back through a racy guard-liveness check. Do
   not kill that recovery controller. An abrupt OS or power loss in the interval between provider create
   and guard start remains a residual; before any subsequent launch, authenticate
   the provider inventory and manually destroy the recorded run label if present.
4. Writes `bench/runs/<UTC date>/<run-id>/receipt.json` -- `commit_sha`,
   `branch` and `dirty_tree` read from git, validated against
   `docs/run-receipt-schema.json` before every write -- and appends
   `bench/runs/ledger.jsonl`, on success, on a failed command and on
   refusal. `--role`, `--agent`, `--work-id`, `--preregistration`,
   `--hypothesis`, `--expected-result`, `--success-criteria` and
   `--failure-criteria` have no defaults; `decision` is `pending <work-id>`
   (this document's vocabulary: `adopt|refute|void|pending` + link) until a
   reviewer sets it; `cost_usd.actual` is the provider's billing
   (the fake provider bills 0), never a copy of the estimate. A receipt is
   `complete` only with teardown proof (or `not-launched` when refused before
   create). Optional `E4B_SLACK_WEBHOOK` posts `LAUNCHED` / `DONE` /
   `TORN DOWN` / `REFUSED`.

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

Live Vast/RunPod API adapters behind `E4B_RENT_LIVE` (the dry-run fake provider
and the controller-side guard are in-tree; a live create still refuses until
that adapter is armed). Policy numbers, gates, thresholds and claims are not
moved by the launcher.

---

**Status**: Policy numbers confirmed by Jordan on 2026-09-06T04:55Z.
**Machine-readable policy**: [`compute-policy.json`](compute-policy.json)
**Receipt schema**: [`run-receipt-schema.json`](run-receipt-schema.json)
**Ledger location**: `bench/runs/ledger.jsonl`
**Check script**: `scripts/check_run_ledger.py`
