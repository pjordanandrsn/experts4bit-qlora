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

These files contain carriage returns (progress bars), and the repo's CI accepts LF line endings only. They are left out here rather than changed, so every file in this directory is byte-identical to the box's. Each is listed with its sha256; the bytes are in the private receipt store. Store only (contains CR; the repo's LF rule):

- `logs/fetch_bf16.log` `b38cf1bcd5546d1db62725448f56492980386ae12a52e3c3e084f123678c4992`
- `logs/fetch_fp8.log` `b1ebc81c656dbcbc72919097dbe695c13b55add82983eb366ce54436fe18e2bc`
- `logs/fetch_int4.log` `3ed25aa959c8fb00282a06103d36f3af6b00633e85aef0cbb62ad3ba3b4d783e`
- `logs/server_E-hyb.log` `b0fabd7062a9a317e26f3a78d13721a75fc69c51619c8f27673270de01ebc36a`
- `logs/server_S4.log` `b0bfdaf1eeabb35ccea0cb52857549986c71f35c087ce3a48758334fbb633bae`
- `logs/server_S4_instr.log` `c14a3d938951994ba2d13dc4b916ca7e3d0a5c58f378f2506d47e7ffa2db9792`
- `logs/server_S8.log` `cf57cc49dd5a58fa92a318f7691e3b839e2809b41703129d83db8479bf1ac2de`
- `logs/server_S8_instr.log` `7aee5cb086073dcd5cbe4b0f32f44567d0a8a6436a57eefddfbff49a26c108d1`
- `logs/server_V8.log` `22c1de975a3f20afb85fe9631b63d53172ac7add09ad2613b11246a0438cd181`
