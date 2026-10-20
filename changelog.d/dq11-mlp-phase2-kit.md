### DQ11 MLP-adapter BUILD Phase 2 kit (#1083): the fused MLP-adapter update against the unfused one, on one RTX 5090

Phase 1 read `PHASE1_PASS` on the A2000 (`dq11-mlp-phase1-a2000-2`). Phase 2 reads the matched update on the card a GAIN
would license, by Amendment 5 (`bench/dq11/DQ11-MLP-BUILD-AMENDMENT-5.md`).
- **(b) at the science lock** over distinct paths per tensor. The leveled K-split families R1(c), R6(c), R9(c), R3(c),
  R4(c) and R5 are each split in its op's own dtype, before the cast the path already has. With every split off the
  reference is PEFT's path bit for bit, and Phase 1's own redraws are reproduced bit for bit.
  - Landing is judged per tensor, and calibration runs only at k >= 9. Below that a tensor is `UNJUDGED_AT_K`, and the
    read is capped at NO_GAIN.
- **The timed arms:** F0 and F1 ABBA twice, 40 updates each, with T_arm the median of updates 6..40. H is the minimum
  over the four block pairs, against 0.05, with NOISY self-pairs a VOID and a strike, and K8 for quality.
- **Drift:** two untimed arms under configuration C, REPORTED only.
- **The order of the rungs:** engagement, then NOISY, then ACCURACY_FAIL, then CALIBRATION_INDETERMINATE, then
  UNJUDGED_AT_K, then GAIN or NO_GAIN. The first rung that applies decides.
- **Box:** the census flow's bootstrap and prepare by file. The guard fit runs before the install, and the guard is 2 h.

Files: `bench/dq11/dq11_mlp_phase2.py`, `dq11_mlp_phase2_run.py`, `dq11_mlp_phase2_box.sh`, `dq11_mlp_phase2_drive.sh`
and `dq11_mlp_phase2_manifest_template.json`, with `tests/test_dq11_mlp_phase2*.py`.

`bind_manifest` also declares `lane_record.min_disk_gb` 123: the box script's own free-disk floor (120,000,000 KiB from
`df -Pk`, 122.88 GB) rounded up, which the shared launcher's preflight disk check reads.
