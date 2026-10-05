### Tests: the chunked-LM-loss gradient bound gains a model-scale floor (tests only)

- `tests/test_chunked_lm_loss.py` held each gradient tensor to 1e-5 of its OWN largest element. On `qwen3_5_moe` the Gated
  DeltaNet's `dt_bias` / `A_log` gradients peak near 2e-5, against a model maximum near 0.2. The chunked loss's fp32
  reorder reaches them from upstream at the large gradients' scale, so a Linux CI runner read `dt_bias` 1.3x over that bound
  (1.79e-10 absolute; #1159's run 37323695463, outside #1159's diff). The Mac reads 2e-6 of it at every thread count.
- Each tensor is now held to 1e-5 of the larger of its own maximum and 1e-3 of the model's largest gradient. That is a 2e-9
  floor here. A 0.1 % error in `dt_bias`'s gradient is still caught (mutation-checked); a 0.01 % one no longer is, on
  tensors whose whole gradient is 1e-4 of the model's. No package code changes.
