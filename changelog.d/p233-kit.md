### P233 kit: does serving at the solver placement keep quality? (`E4B_PAGED_PLACEMENT=solver` against `all-vram`)

The registration's code PR, no box yet. P232's kit by counted edits.
- **Runner** (`bench/p233/p233_run.sh`): the refusals (CUDA 18, torch image 28, card 15, disk below 140 GB 13, host RAM
  below 60 GiB 16), P231's download probe (40 MB/s, rc 19) and the pre-install guard fit (FIT 8,500 s x 1.25, rc 41)
  before any install; no power floor (the limit is recorded, never ruled); the card's GPU UUID is never collected; e4b at the launch commit and grouped-nf4-gemm
  v0.50.0, with a tripwire on the head's explicit knobs, K36's quantizer and K37's plan, and eager decode at both
  placements; the premise on the card (the tier's tests 37 and the solver composition's 6 passed, rc 25); then the three
  phases A (all-vram), S (solver at 8.0 / 3.0 GiB) and C (solver at 8.0 / 0), every arm with `E4B_PAGED_GRAPHS=0`,
  `E4B_INT8_HEAD_BW=1` and `E4B_INT8_HEAD_ROWS=8`.
- **Box** (`p233_box.py`): P115's `measure_phase`, unmodified and with exactly the keywords `p115_quality.main()` passes,
  at group T = 1 and 8 in one process; beside it the placement proof (experts and bytes per tier, resident bytes, each
  layer's static path), the path counts per T (collapse and tiered calls, DRAM rows on the CPU or the GPU), S's and C's
  argmax and the positions differing from A's, the arena's sha, the revision, the resolved levers, the host CPU with its
  ISA flags and the CPU kernels' thread count.
- **Reducer** (`p233_reduce.py`): VOID (the placement proof per arm included) / per-T SANE / SOLVER_SANE or
  SOLVER_NOT_SANE, and the reported TRIPWIRE (a zero or absent A floor falls to 0.002); a `p233-shake-N` run reads
  SHAKEDOWN. No real fixture yet: the shakedown's record becomes one by amendment.
- **Drive** (`p233_drive.sh`): pod-launch's contract (the launch commit from the clean HEAD, `E4B_SHA` refused, the box
  tree fetched into `$RUN_DIR/p233/` with `--no-links`, a non-empty target refused); the premise staged under `t/`.
- **Tests** (`tests/test_p233.py`): the pins, the PREREG's constants and fit, the runner's order, the drive's refusals
  and fetch, arm A's call proven equal to `p115_quality.main()`'s (its AST and a captured run), the box against a
  committed P232 quality record, the int8 head resolving on at `solver` on CPU, and the reducer and box mutants.
