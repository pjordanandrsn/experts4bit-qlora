# P129 default flip: CUDA correctness smoke (RTX A2000 12GB)

Correctness only. No timing from this card is a claim.

- **Environment.** transformers 5.18.0, bitsandbytes 0.50.2, torch 2.11.0+cu128, grouped-nf4-gemm `d1f64ba` (the DEFAULT_ON boxes' build). `E4B_TRAIN_FUSE_QKV` unset throughout unless a check sets it.
- **Tests** (2026-10-10, 07:10–07:14 UTC, tree `3e03be8`, which carries the package code this PR ships): 416 passed, 2 skipped, 1 failed (`pytest-summary.txt`). The files run:
  - `test_train_qkv_fuse.py`, `test_qkv_fuse.py`, `test_moegen_structural.py` and `test_topology_recipe.py`;
  - `test_absmax_dq.py`, `test_batched_train.py`, `test_chunked_lm_loss.py`, `test_ckpt_offload.py` and `test_epilogue_contract.py`;
  - `test_fast_grouping.py`, `test_fast_v4.py`, `test_fused_train_parity.py`, `test_rope_train.py` and `test_offload.py`;
  - `test_dense_offload_trainable.py`, `test_tc1_arm.py` and `test_tp4_arm.py`.

  The failure, `test_fast_v4.py::test_gptoss_inside_lora_is_still_skipped`, fails the same way at the branch's base `d7ba80da`, on the same card. gpt-oss's `ExpertsLoRA` epilogue contract refuses at construction, and no attention module is involved.
- **Smoke.** `bench/p129/instrument/p129_default_smoke.py` runs the 2-layer Qwen3-MoE (real dimensions) set up as TC1's e4b arm, with fp32 and bf16 adapters.
  - `smoke-run1.json` (07:14 UTC, tree `3e03be8`): every check held except "round trip bit for bit (loss, every adapter grad)". The loss matched bit for bit; the gradients did not.

    A control on the unchanged path explains it. Two identical `E4B_TRAIN_FUSE_QKV=0` builds, each `enable_fast_train` then `disable_fast_train` and one step, also differ in adapter gradients, with largest relative differences 0.0038 (fp32) and 0.0065 (bf16). The reference MoE backward accumulates with atomics. The round trip differed from the off side by the same amounts.
  - `smoke-run2.json` (07:17–07:18 UTC, tree `2ff5f29`, same package code): the round-trip check now reads the forward bit for bit (loss and logits). It compares gradients with the off side's spread, measured in the same run on a second off build. **PASS** on both adapter dtypes:
    - the default fuses both layers and refuses none, and `=0` fuses nothing;
    - each of three AdamW steps' losses is within 1.9e-4 relative of the off side's;
    - `disable_fast_train` restores both layers;
    - the round-trip forward is bit for bit, and its largest gradient difference equals the off side's own;
    - a re-enable fuses both layers, and its step-0 loss is bit for bit the default's.
