### A gpt-oss expert-LoRA oracle, matched to transformers forward and backward

`tests/gptoss_lora_oracle.py` states the function a trained gpt-oss expert must compute: an unfused, bias-carrying
reference with the clamped GLU and the de-interleaved layout `from_gptoss` loads, with `ExpertsLoRA`'s adapter
tensors. `tests/test_t3_gptoss_lora_oracle.py` matches it to transformers' `GptOssExperts` in fp32 to 1e-5, forward and
backward (every LoRA gradient by the chain rule), on one stack and on a 2-layer `GptOssForCausalLM`. It also names
what stands between the oracle and trained gpt-oss experts today: `ExpertsLoRA`'s bias refusal, the clamped GLU not
exposed as `_apply_gate`, no bias on the fused training route, and the interleave's row permutation. Tests only; no
product path changes.
