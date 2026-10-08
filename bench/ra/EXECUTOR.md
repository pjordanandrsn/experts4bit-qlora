# RA executor work

Work item [#1363](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1363).
The registration is merged. These are the first executor components; no GPU
proof or launch clearance is claimed.

`ra_stage.py --repo ABS --pins ABS/source-pins.json --out ABS/instruments`
reads the registered Git tree, checks all instrument hashes, and stages their
local import closure without editing the source. The flat directory preserves
the helpers' sibling imports. Its manifest distinguishes registered files from
additional imports, with hashes for both. `--verify ABS/instruments` refuses
changed or unaccounted bytes. Freeze this manifest's digest in the launch plan.

`ra_quality.measure_r` calls P115's `measure_phase` for its own wikitext R,
captures the original score helper's argmax IDs and checks that its NLL scores
did not change. It restores the hook on success or failure. The measured model
must already have been built with the release's defaults.

`ra_normalize.py --dir ABS/run` constructs four envelopes from retained native
records, checksums the whole directory and applies the registered reducer.
`--check` recomputes the projection and refuses an edited envelope or binding.
The run holds `instruments/`, `run.json` and `raw/<ABBA tag>/` with:

- `metadata.json`, `training.json`, `training_profile.json`, `decode.json`;
- SC2 driver `capacity.json`, `capacity_warm.json`, `capacity_burst.json`;
- `/health` snapshots `capacity_health_warm.json`, `capacity_health_burst.json`,
  `capacity_health_end.json`, plus `capacity_extra.json` (SLO and feature telemetry);
- P115 `quality_1.json`, `quality_12.json`, including captured argmax IDs and
  observed feature telemetry.

The normalizer verifies HTTP status, stream identity, usage and request order.
It regenerates SC2's plans through the staged helper (point seed 112, warm 999,
burst 998 at 1000 req/s), derives counters from `/health`, and requires the
widest bucket's replay count to increase during the burst. It preserves native
training/decode samples and quality scores. Feature/import telemetry still
needs independent collection by the executor; equality with caller-provided
metadata is not GPU engagement evidence.

Remaining before proof clearance: real process wrappers and feature-counter
mapping; wheel/import and input byte verification; exact fixture environment
allowlist; ABBA supervisor; collected on-card premise tests with no skips;
locked dependencies; reviewed launch/ledger/deadline gates; receipt fetch and
checksum verification before teardown, billing and rental-absence proof.
No rental is authorized by these CPU checks.
