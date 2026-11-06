# pgx-shake-r1: the PGX shakedown that served (SHAKE_OK, never read)

The retry box ran Amendment 2's runner and finished 0 with `SHAKE_OK` (`shake_verdict.json`): all four arms served at
8 GB, with 0 faults and every counter shim on a real record. Amendment 3 fits the reading's box plan from these files
(`bench/pgx/pgx_schedule.py`, the `SHAKE` constants):
- the setup wall, STARTED to the first lifetime, from `outer.log`;
- the c1 pass walls and TPOT from `shake/<arm>/8/d1/P1.json` and `P2.json`;
- the lifetime walls and peaks from `shake/<arm>/8/d1/lifetime.json`;
- the graphs-off instrument passes from `instr/<arm>/I1.json`;
- the link from `calib.json`;
- the VRAM mass shares from `place_preview.json`.

These are the files the driver fetched (`pgx_drive.sh`), minus the staged copies of this repo's own code, hooks and prompt
sets (they are the launch commit's files, named in `summary.txt`'s `KNOBS` line) and the NF4 arena's 5 MB manifest. The
provider's instance and machine records stay in the private receipt store. `SHA256SUMS` lists every file here.

## Store only

`logs/server_V8.log` (sha256 `22c1de975a3f20afb85fe9631b63d53172ac7add09ad2613b11246a0438cd181`) has CRLF line endings, which the repo's CI rejects (`scripts/check_crlf.py`). It is left out here rather than changed, so every file in this directory is byte-identical to the box's; the bytes are in the private receipt store. Store only (contains CRLF; the repo's LF rule). Nothing reads it: V8's counters reach the reducer through `shake/V8/8/d1/lifetime.json`.
