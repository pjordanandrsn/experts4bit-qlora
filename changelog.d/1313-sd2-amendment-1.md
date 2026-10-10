### SD2 Amendment 1 (#1313): the CUDA proof `sd2-prove-1` -- its target and its harness

- **The target.** `bench/sd2/PREREG-sd2.md` Amendment 1 registers the correctness-only proof of SD2's stacked build on
  one RTX 5090. The target is integration commit `539a2d26` (main plus build PRs #1553, #1554, #1556, #1558 and
  #1559 at their reviewed heads), with grouped-nf4-gemm v0.45.0.
- **What runs.**
  - the target's own GPU tests;
  - the census and the capture's bitwise oracle;
  - V0's addressing gate at k = 1..3 and the three addressing mutants that must fail it;
  - the draft gate against SD1's reference;
  - the batching transition on the real target.
- **The harness.** `bench/sd2/` (`sd2_run.sh`, `sd2_drive.sh`, `sd2_box.py --prove`, `sd2_reduce.py --prove`,
  `staged.sha256`), `tests/test_sd2.py`, and the dry run `tests/test_sd2_dryrun.py`.
- **The ceiling** is $1.35, the worst case at the policy caps. The read's harness is Amendment 2's.
