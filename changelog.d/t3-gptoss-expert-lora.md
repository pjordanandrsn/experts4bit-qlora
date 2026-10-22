### gpt-oss expert LoRA on the unfused reference loop (opt-in)

- `GptOssExperts4bit` exposes its clamped GLU as `_apply_gate`. Its own forward calls the hook, and stays bit-identical.
- `ExpertsLoRA` wraps a base that has an `_apply_gate` hook and whose only bias tensors are the buffers
  `gate_up_bias [E, gate_up rows]` and `down_bias [E, hidden]`. It adds each bias right after the matching base
  projection, in both the expert-mask forward and the single-token decode forward. The down bias sits inside the
  routing weight, as in the base.
- `assert_stock_epilogue` gains `allow_bias=False`. Only the `ExpertsLoRA` constructor passes `True`. Every other entry
  point keeps refusing a biased stack, because none applies a bias: `enable_fast`, `enable_fast_train`,
  `enable_batched_train`, the hybrid and NVMe training enablers, the loader's `arena_train=True` branch, and, newly,
  `enable_expert_offload` on a wrapper.
- The loader still builds gpt-oss experts bare by default. With `E4B_GPTOSS_EXPERT_LORA=1` it wraps them in
  `ExpertsLoRA`. The opt-in refuses `offload` (`ValueError`). Its cost against a fused route is unmeasured.
- `tests/test_t3_gptoss_expert_lora.py` grades the wrapper against the oracle in `tests/gptoss_lora_oracle.py` on CPU,
  to 1e-5 in fp32: forward and backward on bf16 and NF4 storage, the decode forward, and a 2-layer `GptOssForCausalLM`.
