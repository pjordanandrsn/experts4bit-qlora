### P129 Amendment 3 (#835): the step-0 clause against a floor measured on the box

- **The box.** Token `qwen3fqkv3`: Amendment 2's box, with each q0 arm also computing the step-0 held-out per row under registered bf16
  schedules of the q/k/v base projections. They are A0 (a self-check), D1 (fp32, the reference), D2 (q split in two), D3 (k and v as one
  matmul) and D4 (q split in four).
- **The clause.** e_x is the mean over rows of |x − D1|. The fused path passes when e_B ≤ max(e_A, e_D2, e_D3, e_D4).
- **Unchanged.** Everything else as Amendment 2. `tc1-5090-146` stays QUALITY_FAIL.
- **Calibration.** On the A2000 real-model data the rule passes and the old clause fails.
- **Predictions.** e_B / max in [0.35, 1.10], about 0.65; step 0 passes per arm with about 80 % probability.
