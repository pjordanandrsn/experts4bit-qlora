# Provenance contract

The atomic locks, job-local outputs, stamped docs, and post-hoc controller aggregation are part
of the scientific apparatus, not plumbing. This contract states what a job must produce before its
result may back a documented claim, and how the controller classifies what it finds. A result
without sufficient provenance can be useful for debugging; it must not support a doc claim.

## Execution model (recap)

One controller (owns code, docs, manifests, aggregates), N workers over a shared volume. A worker
claims a job with an atomic `mkdir` on `<locks-root>/<job_id>.lock/`, runs it, and writes ONLY
inside its own `<jobs-root>/<job_id>/`. Workers never append to shared aggregates; the controller
aggregates after reading job-local files. Full rules: `docs/RUNPOD_DISTRIBUTED_VALIDATION.md`.

## As-built job-local files

The idealized plan named `manifest.json` / `environment.json` / `metrics.jsonl`; the runner
(`scripts/runpod_claim_and_run.py`) writes the equivalent set below, and the validator checks
these names:

| as-built file | role | idealized name |
|---|---|---|
| `command.sh` | exact command that ran (manifest-of-record, with the shared manifest line) | manifest.json |
| `env.json` | environment record (pod id, host, pid, commit, cache env) | environment.json |
| `status.json` | running/pass/fail + timestamps | status.json |
| `result.json` | the result row: params + attribution + metrics | result.json |
| `run.log` | merged stdout/stderr | stdout/stderr.log |
| `result_rows.jsonl` | train/query payload rows (runner lifts the last into result.json) | metrics.jsonl |
| `adapter*/` + `expertsnbit_adapter_metadata.json` | train jobs: adapter + provenance sidecar | adapter_provenance.json |
| `profile.jsonl` | expert-streaming jobs: per-layer/expert trace | expert_events.jsonl |

Expert-streaming aggregates (`concentration_summary`, `pinning_simulation`, `policy_candidates/`)
are **controller** outputs from the summarizer, not worker-written — kept out of job dirs.

## What the controller validates before summarizing

`scripts/validate_job_provenance.py` reads each job dir (never mutates it) and checks:

- `result.json` exists and parses; `job_id` inside matches the directory;
- `status.json` (if present) agrees with `result.json` on status;
- required attribution present and non-null: `job_id`, `status`, `commit`, `pod_id`, `hostname`,
  `torch_version`, `started_at`, `finished_at`;
- environment completeness: a GPU name, `cuda_version`, `bitsandbytes_version`, plus `env.json`
  and `command.sh` on disk;
- **integrity**: where the manifest and the result both declare `storage_mode` / `offload` /
  `seed` (and the query equivalents), they must match — a job that ran a different mode than it
  claimed cannot back a claim about that mode.

## Classification

| class | meaning | may back a doc claim? |
|---|---|---|
| `claim_usable` | complete, self-consistent, fully attributed | yes |
| `debug_only` | ran and parseable, but a provenance/integrity check failed (missing commit/GPU/versions, status≠pass, or a manifest-vs-result mismatch) | no — debugging only |
| `invalid` | status/result present but unparseable or internally contradictory (e.g. job_id mismatch) | no |
| `missing` | claimed or listed in the manifest but no result.json | no |

The validator exits nonzero if any manifest job is `missing` or `invalid`; `claim_usable` and
`debug_only` are acceptable end states (the latter simply doesn't feed a claim). Run it, then the
summarizers, then promote observations by evidence level — never summarize an unvalidated tree.

## Claim ladder

Every promoted statement carries its evidence level:

- one run → **observation** ("observed in this run");
- repeated across seeds → **OLMoE-supported** (still model/host/dataset specific);
- a Qwen3 run → **scale-transfer observation** (topology, not inherited OLMoE numbers);
- multiple model families → a future **general** claim.

One-run results are not promoted to general claims. The transferable claim is the topology, not
the exact row winners.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T09:57:30Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `c51b7318aec876f998e8c046f7057a6272936a150c3392edc960efadb4d1ca6b` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[&O:@=~:*%?&*=0$#]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|  .o+     .      |
|  +..=   o o     |
| . = .o . B .    |
|    =. = = =     |
|   .oo& S o      |
|   o+OoB =       |
|   o*=. + .      |
|   oEo           |
|   ....          |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info provenance_contract.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify provenance_contract.md.ots provenance_contract.md` succeeds against the on-disk bytes.
- Anchor file: `provenance_contract.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
