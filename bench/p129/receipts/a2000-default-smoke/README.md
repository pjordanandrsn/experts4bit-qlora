# P129 default flip: CUDA correctness smoke (RTX A2000 12GB)

Correctness only. No timing from this card is a claim. Environment: transformers 5.18.0, bitsandbytes 0.50.2, torch 2.11.0+cu128,
grouped-nf4-gemm `d1f64ba` (the DEFAULT_ON boxes' build). `E4B_TRAIN_FUSE_QKV` is unset throughout unless a check sets it.

The PR's package code changed once after the first runs. The fusion first released the q/k/v bases; it now keeps them registered,
re-pointed at the fused copy (views of the fused bytes, a non-nested quant state over the expanded fp32 absmax). Runs 1 and 2 are
the release design; runs 3 onward are the re-point design.

## Tests

The 17 test files that reach `enable_fast_train` or the fusion:

- `test_train_qkv_fuse.py`, `test_qkv_fuse.py`, `test_moegen_structural.py` and `test_topology_recipe.py`;
- `test_absmax_dq.py`, `test_batched_train.py`, `test_chunked_lm_loss.py`, `test_ckpt_offload.py` and `test_epilogue_contract.py`;
- `test_fast_grouping.py`, `test_fast_v4.py`, `test_fused_train_parity.py`, `test_rope_train.py` and `test_offload.py`;
- `test_dense_offload_trainable.py`, `test_tc1_arm.py` and `test_tp4_arm.py`.

| run | tree | design | result |
|---|---|---|---|
| 1 | `3e03be8` | release | 416 passed, 2 skipped, 1 failed (`pytest-summary-run1.txt`) |
| 3 | `4f81b9db` | re-point | 416 passed, 2 skipped, 5 failed (`pytest-summary-run3.txt`) |
| 4 | `f406e489` | re-point (tests reworked) | `test_train_qkv_fuse.py` and `test_qkv_fuse.py`: 36 passed (`pytest-summary-run4.txt`) |
| 5 | `2dfb5cfb` | re-point (+ the checkpoint-rebuild test) | `test_train_qkv_fuse.py` and `test_qkv_fuse.py`: 37 passed (`pytest-summary-run5.txt`) |

- **The gpt-oss failure** (`test_fast_v4.py::test_gptoss_inside_lora_is_still_skipped`), in runs 1 and 3, fails the same way at the branch's base `d7ba80da` on the same card. gpt-oss's `ExpertsLoRA` epilogue contract refuses at construction, and no attention module is involved.
- **Run 3's other four failures** were the new state-dict tests as first written: they asserted strict loads. torch's `load_state_dict` reports a bitsandbytes 4-bit module's quant-state keys (`weight.absmax`, `weight.quant_map`, ...) as unexpected, including an unfused model's own dict and the never-fused `o_proj`'s. So `strict=True` refuses every 4-bit dict, and a resume loads with `strict=False`, as the Hugging Face Trainer does.
- **Run 4.** The tests now hold a fused model's dict to the unfused baseline: nothing missing, and the same unexpected quant-state keys less the q/k/v nested-statistics pair. They pass. The other 15 files and the package code are unchanged since run 3.
- **Run 5** adds one test: a checkpoint saved while fused (each q/k/v absmax in fp32) rebuilds through `QuantState.from_dict` as a plain quant state that dequantizes bit for bit as the original nested one. The package code is the same as in runs 3 and 4.

## Smoke

`bench/p129/instrument/p129_default_smoke.py` runs the 2-layer Qwen3-MoE (real dimensions) set up as TC1's e4b arm, with fp32 and bf16 adapters.

- **Run 1** (`smoke-run1.json`, 07:14 UTC, tree `3e03be8`, release design): every check held except "round trip bit for bit (loss, every adapter grad)". The loss matched bit for bit; the gradients did not.

  A control on the unchanged path explains it. Two identical `E4B_TRAIN_FUSE_QKV=0` builds, each `enable_fast_train` then `disable_fast_train` and one step, also differ in adapter gradients, with largest relative differences 0.0038 (fp32) and 0.0065 (bf16). The reference MoE backward accumulates with atomics. The round trip differed from the off side by the same amounts.
- **Run 2** (`smoke-run2.json`, 07:17–07:18 UTC, tree `2ff5f29`, release design): the round-trip check now reads the forward bit for bit (loss and logits). It compares gradients with the off side's spread, measured in the same run on a second off build. **PASS** on both adapter dtypes:
  - the default fuses both layers and refuses none, and `=0` fuses nothing;
  - each of three AdamW steps' losses is within 1.9e-4 relative of the off side's;
  - `disable_fast_train` restores both layers;
  - the round-trip forward is bit for bit, and its largest gradient difference equals the off side's own;
  - a re-enable fuses both layers, and its step-0 loss is bit for bit the default's.
- **Attempts at the re-point design** (07:56–08:01 UTC): out of memory while loading a model. Other services on the card held about 8.2 GB, leaving about 3.6 GB. The smoke now collects each model's reference cycles before the next build, and keeps the compared logits on the host. Its checks are unchanged.
- **Run 3** (`smoke-run3.json`, 08:41 UTC, tree `2dfb5cfb`, re-point design, run once the card had room): **PASS** on both adapter dtypes:
  - the default fuses both layers and refuses none, and `=0` fuses nothing;
  - each of three AdamW steps' losses is within 1.9e-4 relative of the off side's;
  - `disable_fast_train` restores both layers;
  - the round-trip forward is bit for bit (loss and logits), and its largest gradient difference equals the off side's own (0.0038 fp32, 0.0065 bf16);
  - a re-enable fuses both layers, and its step-0 loss is bit for bit the default's.
