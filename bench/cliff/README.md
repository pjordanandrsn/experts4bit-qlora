# The W1/W2/W4 cliff census (EXPLORE tier)

**Not a reading.** Run under the EXPLORE tier (adertha-agents#348): no preregistration, never quoted as a finding, never
in public text, never used to flip a default. If it names something worth claiming, that claim gets its own
registration.

## Question

Today a 2-row decode step costs about twice a 1-row step (SD2 V1, measured on e4b 0.53.0, before K39's decode and the c1
plan). Where does the W2..W4 step's extra time go on **today's default server**, kernel by kernel: the expert GEMMs,
the routes that exist only at one row (the int8 head GEMV, the dense GEMVs), attention, or the glue?

## Instrument

P152's per-name census (`bench/p152/`, itself P145's) by counted substitutions:

- **Batches 1, 2 and 4**, not 1 and 16.
- **grouped-nf4-gemm v0.50.0** (`41ef086e`), the release a `pip install` of e4b resolves to today, not v0.47.1.
- **The speed arms** run `cliff_box.py`: P136's box with P109's workloads widened to W1, W2, W4 and W16, so every census
  batch has its own unprofiled step. The census divides by that step, never by the profiled period.
- **No routing census and no P152 verdict.** `cliff_reduce.py` tabulates each batch by kernel name and by class, then
  writes `removable(W2) - removable(W1)` and `removable(W4) - removable(W1)`, largest first. A name whose grid changes
  with the batch is summed over its grids.
- **The premise passes on rc 0**, because the test files are this commit's.

Everything else is P152's: the host refusals, the 575 W power floor, the exact nsys build, graph- and node-mode captures
of 64 steps after 32 skipped, the per-name table and the class map.

## Running

`cliff_drive.sh` is launched by the guarded launcher (`pod-launch.sh`) on one RTX 5090 with run id `cliff-5090-N`. It
installs e4b at the launch commit (a clean, pushed tree) and grouped-nf4-gemm at the pin above. Results land in the run
directory's `cliff/` (no checkpoint, arena or raw Nsight report travels): `verdict_cliff.json`, `summary.txt`, the arms
and the per-name tables.
