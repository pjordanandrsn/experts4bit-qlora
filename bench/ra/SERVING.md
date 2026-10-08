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

`ra_env.py` constructs a fresh process environment with separate release caches
and rejects extra flags in direct wrapper invocations. Serving fixtures may set
only model/revision, arena/calibration, CUDA/all-vram placement and common thread
count. Quality additionally names graphs=0. Capacity additionally names context
2048, chunk 512 and trace destinations. Training names no feature environment.
No speed, route, bucket, slot or fusion flag is allowed. Removed key names are
recordable; inherited values and credential variables are not copied into logs.

These wrappers are not on-card evidence. Remaining work includes training and
capacity wrappers, complete independently verified feature/fallback coverage,
wheel/input provenance, ABBA supervision, collected no-skip premise tests and
reviewed launch/receipt-retrieval/teardown gates. No proof clearance requested.
