### P232 kit: the served read of the int8 lm_head at 2, 4 and 8 rows (`E4B_INT8_HEAD_ROWS=8`, lever K, K2b)

The registration's code PR, no box yet. P153's kit by counted edits, on today's harness.
- **Runner** (`bench/p232/p232_run.sh`): the refusals (CUDA 18, torch image 28, card 15, disk 13, host RAM 16, the
  575 W floor 17), P231's download probe (40 MB/s, rc 19) and the pre-install guard fit (FIT 7,360 s x 1.25, rc 41; Amendment 1)
  before any install; e4b at the launch commit and grouped-nf4-gemm v0.50.0, with a tripwire on the head's grid, its
  qwen3_moe default and K37's shipped plan; P229's premise on the card (11 and 45 passed, rc 25); then the warm-up
  and eight ABBA speed arms at W2, W4 and W8, and the three quality phases (R, K, B16) at group 2, 4 and 8.
- **Box** (`p232_box.py`): speed records the head's engagement at the build and, per workload, the graph replays and
  the head's eager calls; quality runs P115's `measure_phase` once per T in one process (references per T), records
  the head's calls during each T, and reports per-row bucket stability from saved T-row decode inputs.
- **Reducer** (`p232_reduce.py`): VOID / NOISY / FUNCTION_FAIL / per-T PASS / FLIP_ROWS_8, _4, _2 or NO_FLIP, and
  the reported TRIPWIRE (a zero or absent R floor falls to 0.002). A `p232-shake-N` run reads SHAKEDOWN. No real
  fixture yet: the shakedown's record becomes one by amendment.
- **Drive** (`p232_drive.sh`): pod-launch's contract (the launch commit from the clean HEAD, `E4B_SHA` refused, the box
  tree fetched into `$RUN_DIR/p232/` with `--no-links`, a non-empty target refused); P229's premise staged under `t/`.
- **Tests** (`tests/test_p232.py`): the pins (shared pieces at P153's and P229's), the PREREG's constants and fit, the
  runner's order, the drive's refusals and fetch, P153's pieces carried verbatim, and the reducer and box mutants.
- **Amendment 1** (`PREREG-p232.md`, pre-data): what the kit could implement exactly. K against B16 is read by its NLL
  gap only; the one-row route is checked at the speed builds; the tripwire floor is per T; `TRIPWIRE unread` on a VOID;
  argmax compared as exact pooled counts; a missing KL or argmax on the scored arm is a VOID; the shakedown runs 3
  reps; the speed arm need is 240 s, so FIT 7,360 s, guard 2.75 h, worst $5.69; the box refuses below 140 GB free.
