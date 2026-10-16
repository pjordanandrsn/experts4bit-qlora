### Added

- R2-Mixtral's box harness (`bench/fam235/`, #1630), built as Amendment 2 decided: R2's own lock (`model_files.json`,
  Mixtral-8x22B at `cc88a6cc`, 69 files), closure (`r2.sha256`) and tiny Mixtral (`r2_tiny.py`). Thin wrappers run Q235's
  stage and fetch-gate tools unchanged. R2's copies of Q235's runner and drive (`r2_run.sh`, `r2_drive.sh`) are held to Q235's
  line for line by `r2_drift.py`: strict against the cited Q235 commit, with drift against the working tree reported.
  Its Python twin classifies every file a wrapped Q235 module reads, so a new one fails until it is declared.
- The science mode refuses until the amendment that registers the box pins its class, floors and caps. No box, rental or
  claim.
