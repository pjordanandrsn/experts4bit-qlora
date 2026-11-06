# pgx-shake-1: the first PGX shakedown box (never read)

The box ran Amendment 1's runner and ended rc 12 in the bake's first second (PREREG-pgx.md, Amendment 2). This is the record
Amendment 2 rule 6 names: STARTED, `PGX_EXIT_CODE` 12, `logs/bake.log` beginning `BAKE FAILED`, `outer.log` saying
`BAKE FAIL: no arena`, and no `logs/calibrate.log` and no `shake/`.

These are the files the driver fetched (`pgx_drive.sh`), minus the staged copies of this repo's own code, hooks and prompt
sets (they are the launch commit's files, named in `summary.txt`'s `KNOBS` line). The provider's instance and machine
records stay in the private receipt store. `SHA256SUMS` lists every file here.
