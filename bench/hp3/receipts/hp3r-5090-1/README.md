# `hp3r-5090-1`: HP3's (R) read, NOT_READ (host-limited)

One rented RTX 5090 (32,607 MiB, driver 595.91.07, a 400 W power limit, a 3,090 MHz maximum SM clock) on an Intel Xeon Platinum
8160-class host (192 logical CPUs, 96 physical cores, 4 NUMA nodes, a cgroup quota of about 92 CPUs; `host_facts.json`). A Vast RTX
5090 at USD 0.7014/h, 2026-10-24 22:06Z to 2026-10-25 01:23Z, under Amendment 4's 6.0 h guard. Cost USD 2.369 (registered estimate
USD 6.20). Code: e4b `8be4af31` (Amendment 4's merge, experts4bit-qlora#2112), the (R) fallback's pins (`versions.txt`): the e4b
side on torch 2.8.0+cu128 and transformers 5.18.0, the Unsloth side on Unsloth 2026.9.14 with transformers 5.5.0.

**Reading: NOT_READ, host-limited (fewer than 3 seeds fit).** Every pair reads NOT_READ, with a host refusal note. It is no strike.
- **G1' passed** (`tc1/hp3/g1.json`, from `bench/hp3/hp3_reduce.py g1` on `tc1/hp3/eval_cal.json`): the e4b and Unsloth bases
  have the same digest, every zero-adapter row agrees within 1e-05 (the largest difference is 0.0), and the row means are equal.
  The calibration arms (one step per framework, seed 3407) ran first. G1' is the instrument gate only: it is not a reading.
- **The seed plan** (`tc1/hp3/seed_plan.json`, Amendment 1 part 7): it reads wall time and the deadline only. Seed 3410 took
  4,814 s of wall for the two frameworks, which left 9,972 s. Three seeds need 2 x 4,814 + 360 x 2 x 3 + 900 = 12,688 s (the
  900 s reserve included), more than the 9,972 s left, so n = 0: the box stopped after seed 3410, with no evaluator pass.
- **Nothing is read from seed 3410.** Its files stay in the private receipt store, listed by sha256 in `STORE-ONLY.sha256`.

**Re-derivation.** The maintainer re-derived the reading from the private receipt store with `hp3_reduce.py` at `8be4af31`
(equal to the box's copy): `g1` on the stored `tc1/hp3/eval_cal.json` reproduced `tc1/hp3/g1.json` byte-identically, and the seed
plan's arithmetic matches Amendment 1 part 7.

**The host, as observed (not part of the verdict).** Both calibration arms ran with the SM clock flat near 1,490 MHz (busy median
1,485 MHz on e4b's arm and 1,492 MHz on Unsloth's, never above 1,492 MHz), with no clock-reason bit set (`tc1/gpuclk_*_s3407.txt`).
The host's single-thread loop read 16.78 Mops/s. Seed 3410's Unsloth arm reached its 3,600 s alarm. A re-run on the registered
spare id takes a host rule for both, as a reviewed kit edit.

**What is here.** The files that decide the reading, byte-identical to the store: the G1' record and its evaluator output, the
seed plan, the build records, the calibration arms' clock and VRAM samplers and logs, the tripwire, dataset and pip logs,
`versions.txt`, `ds_manifest.json` and `host_facts.json`. The two clock logs each contain one digit run that equals a rented-host
number. Both sit inside `/proc/stat` CPU counters, and neither is an identifier.

**What is not here** (`STORE-ONLY.sha256`, each at the same sha256):
- seed 3410's files and the export manifest that names its adapter;
- the files that carry the rented host's identifiers (the instance id, the container's host name or the GPU's UUID): `box.json`,
  `forensics.txt`, `outer.log`, `summary.txt`, the calibration arm records, `INSTANCE_ID` and the box's TC1 RESULTS copies;
- three logs with carriage returns, the run's markers, the staged copies of committed code, and the dataset and token files.
The launcher's receipt and its ledger row are only in the store.

`SHA256SUMS` lists every file of the box's receipt with its sha256 in the private receipt store.

- **No identifiers:** no file here names an instance, an offer, a machine, a host name, an endpoint or the GPU's UUID.
