### The training estimate prices the absmax the run stores, and tells the loss branch from model_type

- **Absmax, as stored.** Under `enable_fast_train`'s default (experts resident on the GPU, the grouped kernel, NF4/FP4
  at blocksize 64, `E4B_ABSMAX_DQ` unset or `1`), the run double-quantizes the expert absmax. It keeps a uint8 per 64
  weights, plus an fp32 scale per 256, an offset and a 256-entry code, instead of fp32 per 64. The estimate now prices
  those stored bytes, so it falls for those setups: by 1.35 GB on Qwen3-30B-A3B, matching the 1.34–1.35 GB TC1
  amendments 28 and 56 measured, and by 10.6 GB on Qwen3-235B-A22B. Expert offload, the reference loop
  and `E4B_ABSMAX_DQ=0` keep fp32, as before. `estimate_env()` reports the switch.
  - The bytes are pinned against MEASURED ones: `tests/test_estimate_absmax_and_family.py` quantizes real NF4 stacks
    on CPU, compresses them (or not) with the run's own code, and requires the estimator's bytes to equal the stored
    tensors' bytes exactly, including counts that are not a multiple of 256.
- **The loss branch.** Whether the run chunks its LM loss was read from `config.architectures` alone. A config built
  without that field (for example `AutoConfig.for_model`) silently priced the stock loss: 1.65 GiB more on
  Qwen3-235B-A22B at T = 2048. It now falls back to `model_type`, as transformers does when it instantiates the class.
  Where neither identifies the family, the stock (larger) loss is priced and the estimate says so in `unmodelled`.
