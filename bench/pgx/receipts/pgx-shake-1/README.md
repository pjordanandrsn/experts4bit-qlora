# pgx-shake-1: the first PGX shakedown box (never read)

The box ran Amendment 1's runner and ended rc 12 in the bake's first second (PREREG-pgx.md, Amendment 2). This is the record
Amendment 2 rule 6 names: STARTED, `PGX_EXIT_CODE` 12, `logs/bake.log` beginning `BAKE FAILED`, `outer.log` saying
`BAKE FAIL: no arena`, and no `logs/calibrate.log` and no `shake/`.

These are the files the driver fetched (`pgx_drive.sh`), minus the staged copies of this repo's own code, hooks and prompt
sets (they are the launch commit's files, named in `summary.txt`'s `KNOBS` line). The provider's instance and machine
records stay in the private receipt store. `SHA256SUMS` lists every file here.

## Store only

These files contain carriage returns (progress bars), and the repo's CI accepts LF line endings only. They are left out here rather than changed, so every file in this directory is byte-identical to the box's. Each is listed with its sha256; the bytes are in the private receipt store. Store only (contains CR; the repo's LF rule):

- `logs/fetch_bf16.log` `aa711c33a75c05f2db67f8680d1836432ce1429a76af6fbc5bbe07a64aa7cd51`
- `logs/fetch_fp8.log` `40d1b581393da0fb731342311920304b405e07871017550ce73934fce9136c53`
- `logs/fetch_int4.log` `d7f00a5c0c57063df35d826c578be597dbeb1796ebc7c5335b99fedd1643ddaa`
