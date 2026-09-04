# Docs index — what each document is, and whether it is current

Forty-two documents accumulated over two months of measured campaigns.
This index says what each one is *for*, whether it is still the thing to
read, and whether it is OpenTimestamps-anchored (**anchored** documents
are never edited in place — they are superseded or archived, and their
`.ots` proof must keep matching their bytes).

Start here: [`STATUS.md`](STATUS.md) (one page, current) →
[`claims.json`](claims.json) (every number with its evidence) →
[`METHODOLOGY.md`](METHODOLOGY.md) (how each number was made).

## Current — read these

- **[`SOLUTIONS.md`](SOLUTIONS.md)** and **[`solutions/`](solutions/)** —
  routing/usage: one page per ordinary problem (symptoms, cause, install,
  smallest example, verification, limits, evidence by claim ID). Current.
- **[`capabilities.json`](capabilities.json)** (+ [`capabilities.schema.json`](capabilities.schema.json)) —
  the machine-readable capability contract; validated in CI by
  `scripts/check_capabilities.py` against pyproject, source and the claims
  register. Current.
- **[`system-manifest.json`](system-manifest.json)** — the cross-repository
  system manifest: what each of the two packages owns, the compatibility
  records (which version of this package needs which grouped-nf4-gemm
  floor, and why), the evidence vocabulary, the invariants. Byte-identical
  in both repositories; validated by `scripts/check_system_manifest.py`
  against pyproject, the claims register and `capabilities.json`. Current.
- **[`change-impact.json`](change-impact.json)** — which files must move
  together (public API, new kernel capability, measured result, dependency
  floor); `scripts/check_change_impact.py --base <ref>` checks a diff.
  Current.
- **[`discovery-queries.json`](discovery-queries.json)** — the
  discoverability regression corpus (`scripts/check_discovery_contract.py`).
  Current.
- **[`RELEASE_NOTES_GUIDE.md`](RELEASE_NOTES_GUIDE.md)** — how the first
  paragraph of a release note is written. Current.

| doc | what it is |
|---|---|
| [`STATUS.md`](STATUS.md) | what you get today, what was retired, what is open |
| [`claims.json`](claims.json) / [`claims-schema.md`](claims-schema.md) | machine-readable register of every claim, with status and evidence path |
| [`CHOOSING.md`](CHOOSING.md) | which entry point, by the constraint you hit (VRAM, host RAM, disk) |
| [`METHODOLOGY.md`](METHODOLOGY.md) | hosts, protocols, every measurement's provenance; §13 serving parity, §13.1 the routing-flip floor |
| [`SERVING-PARITY.md`](SERVING-PARITY.md) | per-family parity of paged decode against the model's own attention |
| [`SERVING-THROUGHPUT.md`](SERVING-THROUGHPUT.md) | per-family decode throughput (B=1, B=16) under the Qwen3-30B campaign's protocol; refused arms = the build-out list; receipt in `bench/hybrid-g9/throughput-20260904/` |
| [`STORAGE-MODES.md`](STORAGE-MODES.md) | the nf4/fp4/int8/fp8/bf16/fp16 support matrix and what each promises |
| [`RESIDENCY-ENGINES.md`](RESIDENCY-ENGINES.md) | the residency engines, hot-set selection, host-regime laws |
| [`ARCHITECTURE_SUPPORT.md`](ARCHITECTURE_SUPPORT.md) | which model families load, run and capture, with evidence |
| [`DEEPSEEK-V4.md`](DEEPSEEK-V4.md) | V4's storage split, epilogue, arena bake, key mapping |
| [`SERVING.md`](SERVING.md) | the FastAPI shim and Docker deployment |
| [`BENCHMARKS.md`](BENCHMARKS.md) | the benchmark scripts and how to run them |
| [`BITSANDBYTES.md`](BITSANDBYTES.md) | relationship to bitsandbytes, prior art |
| [`OFFLOAD-TRANSFER-NOTES.md`](OFFLOAD-TRANSFER-NOTES.md) | H2D bandwidth diagnostics; defers to METHODOLOGY §12b for the decode grid |
| [`hybrid/PORTABILITY.md`](hybrid/PORTABILITY.md) | naming rules and the per-family onboarding checklist for the serving engine |
| [`provenance_contract.md`](provenance_contract.md) · anchored | what a job must record before its result can back a claim |
| [`RUNPOD_DISTRIBUTED_VALIDATION.md`](RUNPOD_DISTRIBUTED_VALIDATION.md) · anchored | the multi-pod execution protocol |

## Superseded — kept, read with the note

| doc | superseded by / note |
|---|---|
| [`INFERENCE.md`](INFERENCE.md) | its decode grid is v0 offload-path figures; the pipelined and paged engines supersede them for decode (the doc says so). The mechanics and kill-switches are still current. |
| [`support_matrix.md`](support_matrix.md) · anchored | the 2026-07-05 OLMoE/Qwen3 storage-mode matrix. Current for what it covers; serving parity is in `SERVING-PARITY.md`. Its footer's disclosed pre-footer hash no longer matches its pre-footer bytes — an older discrepancy, recorded, not fixed (that would mean editing an anchored file). |
| [`hybrid/INTEGRATION-ASSESSMENT.md`](hybrid/INTEGRATION-ASSESSMENT.md) | §5's recommendation (vLLM hosts the throughput surface) is superseded by its own §6: decided 2026-08-16, own system. Kept as the record of what was evaluated. |
| [`hybrid/ARCHITECTURE-NOTES.md`](hybrid/ARCHITECTURE-NOTES.md) | the Stage 1–2 pre-work map (2026-08-11/19). Superseded in parts by what shipped; read the CHANGELOG for what actually landed. |
| [`hybrid/OBJECTIVE-REVISION-2026-08-23.md`](hybrid/OBJECTIVE-REVISION-2026-08-23.md) | registered predictions P1–P4 for the B=16 campaign; graded in `bench/hybrid-g9/`. |

## Research record — the OLMoE campaign of 2026-07-04 → 07-06 (all anchored)

These are pre-registrations, results, certificates, ledgers and plans
from one three-day campaign on OLMoE-1B-7B (with a Qwen3-30B probe).
They are dated, immutable, and self-correcting — several grade their own
earlier claims as RETRACTED or CORRECTED, and that is the point of them.
None is the current position; `results_summary.md` was the decision
surface *of that campaign*.

| doc | role in the campaign |
|---|---|
| [`results_summary.md`](results_summary.md) | the campaign's decision surface (re-anchored 7×) |
| [`MEASUREMENT_AUDIT.md`](MEASUREMENT_AUDIT.md) | external-review recomputation; supersedes the grid's "best eval" framings |
| [`OLMOE_EXPERTSNBIT_GRID.md`](OLMOE_EXPERTSNBIT_GRID.md) | the six-mode grid, single run per cell; downgraded by the audit |
| [`OLMOE_REPEAT_VALIDATION_PLAN.md`](OLMOE_REPEAT_VALIDATION_PLAN.md) | 3-seed repeats and repeat-5 decode |
| [`QWEN3_30B_EXPERTSNBIT_GRID.md`](QWEN3_30B_EXPERTSNBIT_GRID.md) | the scale-transfer probe |
| [`MODE_DECOUPLED_ADAPTERS.md`](MODE_DECOUPLED_ADAPTERS.md) | train/query storage-mode portability matrix |
| [`TRAIN_PLACEMENT_CERTIFICATE.md`](TRAIN_PLACEMENT_CERTIFICATE.md) | bitwise resident-vs-offload certificate, rev1 → rev3; ends in a scoped open item (S10) |
| [`EXPERT_STREAMING_PROFILE.md`](EXPERT_STREAMING_PROFILE.md) | hot-expert concentration gate — a clean negative |
| [`OFFLOAD_MEMORY_FACTS.md`](OFFLOAD_MEMORY_FACTS.md) | peak-memory arithmetic; retires the "persistent workspace" reading |
| [`LAYOUT_FACTS.md`](LAYOUT_FACTS.md) | source facts vs measurement precedence; full-run determinism UNKNOWN |
| [`DIVERGENCE_ONSET_PROBE.md`](DIVERGENCE_ONSET_PROBE.md) | pre-registered probe + result; its "queue empty" line was corrected by `NEXT_CAMPAIGN_LANES.md` |
| [`NULL_LADDER_1024_AMENDMENT.md`](NULL_LADDER_1024_AMENDMENT.md) | the n=1024 eval-set amendment |
| [`PREDICTION_LEDGER.md`](PREDICTION_LEDGER.md) | append-only ledger of graded predictions, including RETRACTED and FALSIFIED entries |
| [`POST_AUDIT_WORK_QUEUE.md`](POST_AUDIT_WORK_QUEUE.md) | the post-audit queue; quarantines Q1–Q4 and stops S8–S10 |
| [`NEXT_CAMPAIGN_LANES.md`](NEXT_CAMPAIGN_LANES.md) | licensed continuations; corrects the "queue empty" claim |
| [`N1_ROUTING_PINNED_SERVE.md`](N1_ROUTING_PINNED_SERVE.md) | a pre-registration that was never run |
| [`SPECULATIVE_LANES_PLAN.md`](SPECULATIVE_LANES_PLAN.md) + [`ADDENDUM_1`](SPECULATIVE_LANES_ADDENDUM_1.md) · [`_2`](SPECULATIVE_LANES_ADDENDUM_2.md) · [`_3`](SPECULATIVE_LANES_ADDENDUM_3.md) | exploratory lanes, "second-class by construction"; Addendum 3 has an explicit staleness reconciliation |
| [`PLAN_ROUTED_V3_AMENDMENTS_A1-A4.md`](PLAN_ROUTED_V3_AMENDMENTS_A1-A4.md) | amendments to a plan that is not in this repository |
| [`expertsnbit_finish_plan.md`](expertsnbit_finish_plan.md) | the v0.2.x completion plan, with its own calibration correction and the cross-card verification matrix |

## Pre-registrations for later campaigns

| doc | note |
|---|---|
| [`PREREG-flagship-matrix.md`](PREREG-flagship-matrix.md) | unanchored; its ten cells are therefore `measured`, not `confirmed` |
| [`PREREG-flagship-matrix-model2.md`](PREREG-flagship-matrix-model2.md) · anchored | the second model, stamped pre-data and pre-hardware; says the above out loud |

## The evidence layer, outside `docs/`

Receipts for the numbers in `claims.json` live beside the scripts that
produced them: `bench/*/RESULTS-*.md` (74 files), `PROVENANCE.md`
(anchored, the v0.2.0 convergence record specifically), and
`audits/`. Several serving-lane receipts (the P-series) live in a
private audit tree and are marked `measured-private` in the register.
