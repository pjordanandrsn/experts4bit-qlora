# RA decode and quality processes

`ra_serving.py` runs in one selected release venv per invocation, with
`--mode decode|quality --spec ABS.json --instruments ABS --out ABS.json`.
It verifies staged helpers before importing them, requires one RTX 5090/sm_120,
and builds the model with `PagedServeConfig.from_env()` plus `build_engine`.
The supervisor must first verify wheels, imports and input bytes.

The decode spec names `kind`, `prompts`, `short`, `long`, `reps`; the quality
spec names `kind`, `windows`, `group`, `cont`, `ref_dir`. Shapes follow PREREG.
Prompts are P109's prepared 16 rows; quality windows are the prepared wikitext
JSON object. The wrappers do not tokenize independently in old and new venvs.
Separate native outputs feed the receipt normalizer.

Kernel counters install before fusion assembly. QKV hooks install after folds
and before graph capture. Decode records capture dispatch, workload dispatch,
replays, kernel calls and fusion census/modes. Graph replays do not re-enter
Python; enabled features are tied to their capture counts and observed replay.
Wall samples retain the helper's full precision. Quality counts dispatch per R
pass and requires singleton GEMV at group 1, no singleton dispatch at group 12.
It retains the helper's attention/bucket evidence and score-preserving argmax
capture. Temporary assembly/pass/score hooks restore after failures.

`ra_fallback.py` snapshots the initial config and reconstructs it independently.
At fusion assembly it resolves unset modes with the loaded model type using the
release resolver, retaining raw/resolved modes and native sources; legacy releases
retain their direct modes. This avoids mistaking the build's capacity resolution
for a configuration override. Capacity may replace host/port only with a
verified listening loopback socket; other config fields still match the native
factory. It reconciles observed
glue forward coverage with native census and rejects auto kernel gaps. Nine baseline 0.50.0 bodies plus the two main router cast variants and
the three router forwards that hand the kernel unwidened logits (#1313, P127's
Phase 1) carry source commits in the adapter registry. They bind complete native forward bodies; changed
bodies require a reviewed adapter. Private clones observe retained-original
calls without changing installed function defaults. Small-row fallbacks and
large-row prefill fallbacks are separate, with per-pass and per-module counts.
This observes Python eager/capture execution, not replayed graph internals.
`ra_routes.py` binds the QKV, residency and NF4 modules to source-bound AST
adapters. Schema 2 uses explicit canonical AST JSON: empty optional generic
parameters are omitted and all other fields are retained, independent of
Python AST display formatting. Older fingerprints refuse. QKV coverage checks structural candidates, retained projections and
the active fused forward, then reconciles per-projection calls. Its zero-fallback
basis is the complete supported fusion, which retains no unfused projection.
GEMV observes each all-VRAM MoE forward and its retained-reference branch,
requires both singleton projections, and joins native primary dispatches to
the release's default route rules. The bandwidth split-K counter is
supplementary, not a second dispatch. Per-pass census/counter gaps, unknown
module bodies, plan overrides and unexpected routes refuse. Failure snapshots
go to the retained process log; temporary route hooks restore on exceptions.
Python counts still require the separate on-card premise and replay gates.

`ra_env.py` constructs a fresh process environment with separate release caches
and rejects extra flags in direct wrapper invocations. Serving fixtures may set
only model/revision, arena/calibration, CUDA/all-vram placement and common thread
count. Quality additionally names graphs=0. Capacity additionally names context
2048, chunk 512 and trace destinations. Training names no feature environment.
No speed, route, bucket, slot or fusion flag is allowed. Removed key names are
recordable; inherited values and credential variables are not copied into logs.

These wrappers are not on-card evidence. CPU controls use native glue bodies
with CPU kernel stand-ins, not real GPU kernels. Remaining work includes training and
capacity wrappers, broader default/feature integration for training and capacity,
wheel/input provenance, ABBA supervision, collected no-skip premise tests and
reviewed launch/receipt-retrieval/teardown gates. No proof clearance requested.
