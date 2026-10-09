### P129 Amendment 2 (#835): the fused q/k/v projection's speed A/B at TC1's field recipe

- **The box.** Token `qwen3fqkv`: `E4B_TRAIN_FUSE_QKV` 0 against 1 on the shipped (bf16) and matched (fp32) adapter arms, two
  draws a side in ABBA order, every arm profiled, on a host-bound RTX 5090 with the launcher's machine ranking off.
- **The gates.**
  - A recount: the box's own launch cut is at least 0.8 of Phase 1's 14.0 %.
  - A premise: the matched arm's GPU busy share is at most 0.85.
  - Wall and device ratios, read separately.
  - TC1's held-out bars on both arms.
- **The harness.** The receipt gains a `train_qkv` record and `profile.launches_per_step`. `tc1_reduce.py` scores the family
  (`fqkv_why`, `score_fqkv`), with a self-test case for each rung.
- **The predictions.** Launches −12 to −15 % on each arm, and wall `q1 / q0` in [0.86, 0.95] on a host-bound host.
