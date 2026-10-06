### Compatibility: transformers 5.19.0 (released 2026-10-06)

- **The meta-tensor rope rebuild** (`arch/moe_load.py`). transformers 5.19 changed
  `<Model>RotaryEmbedding.compute_default_rope_parameters` to `(config, **kwargs)`, so the positional `device` argument
  raised `TypeError`. The rebuild did not run, which broke streaming loads of a model whose `inv_freq` was still on
  meta, such as Qwen3-MoE. It now passes `device=` by keyword, which every version accepts, and moves the result onto
  the device, because 5.19 returns a CPU tensor.
- **The fused-layout probe's toy config** (`arch/fused_layout_probe.py`) now carries `swiglu_alpha=1.702`. 5.19's
  `GptOssExperts` reads the value from the config where earlier versions hard-coded it. The value equals e4b's
  `GPTOSS_ALPHA`.
- **Tests.** `tests/test_moe_load.py` and `tests/test_fused_layout_probe.py`: 39 passed on transformers 5.17.0 and on
  5.19.0. Before the fix, 6 failed on 5.19.0, the same 6 that failed e4b's CI after 5.19.0 was published.
