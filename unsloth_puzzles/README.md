# Unsloth Puzzles — Task A (Triton NF4) & Task B (FSDP2 + QLoRA)

Developed and verified on a single **RTX A2000 12 GB** (Ampere). Where the puzzle rubric targets a
**Tesla T4** or a **2× T4 Kaggle** box I don't have, that's called out explicitly — numbers here are
what was actually measured, not extrapolated silently.

## A) Convert NF4 to Triton — [`../experts4bit_qlora/triton_nf4.py`](../experts4bit_qlora/triton_nf4.py)

A **single** Triton kernel does the full **double dequant** (bitsandbytes `compress_statistics=True`):
it reconstructs the nested per-64-block `absmax` on the fly
(`state2.code[absmax_u8[blk]] * state2.absmax[blk//256] + offset`) and applies the 16-entry NF4
codebook per nibble — one launch, no intermediate absmax buffer.

| rubric item | status |
|---|---|
| single Triton kernel | ✅ both dequants fused in one kernel |
| correct in f16 **and** bf16 | ✅ `assert_close` vs `bnb.dequantize_4bit` (bf16 bit-exact, fp16 within ULP) — `../tests/test_triton_nf4.py` |
| cache eviction | ✅ `eviction_policy="evict_first"` on the streaming weight load |
| works in `torch.compile` | ✅ registered as a `torch.library.triton_op`; compiles `fullgraph=True` (no graph breaks) — tested |
| custom PTX asm (+3) | ✅ nibble unpack via inline PTX `bfe.u32` (`tl.inline_asm_elementwise`); matches bnb, all tests pass |
| speedup ≥ 1.15× | **measured ~1.3× vs `bnb.dequantize_4bit`** on the A2000 (1.21–1.47× by shape) |

**Honest caveats on the speedup:** the rubric measures vs Unsloth's `fast_dequantize` on a **T4**.
I benchmarked vs **bitsandbytes** on an **A2000** (Unsloth/peft wrap bnb; per the puzzle's own numbers
Unsloth is ~1.05× over bnb → 1.29× vs bnb ≈ ~1.23× vs Unsloth, clearing 1.15×, but that's an
*extrapolation*). `your_dequantize_nf4(module)` drops straight into the puzzle's `test_dequantize` —
run it on a T4 for the exact rubric number.

```bash
pytest tests/test_triton_nf4.py          # correctness (f16+bf16) + torch.compile
```

## B) FSDP2 + QLoRA — [`fsdp2_qlora_sft.py`](fsdp2_qlora_sft.py) + [`fsdp2_config.yaml`](fsdp2_config.yaml)

A single transformers-native script (`SFTTrainer`) launched under an FSDP2 accelerate config. Two
things make FSDP2 + bnb-NF4 actually shard: `bnb_4bit_quant_storage=<bf16/fp16>` (so the packed
weights live in a shardable dtype) and **no** `device_map` (FSDP2 places the model).

**What was verified here (single A2000):** [`_validate_fsdp2.py`](_validate_fsdp2.py) runs FSDP2
`fully_shard` in a single-rank group over a tiny Llama (SmolLM2-135M), 4-bit + LoRA:

```
[fsdp2] wrapped 30 decoder layers + root; offload=CPU, mp=bf16
[fsdp2] trainable=4,884,480 (base-trainable params=0)
[fsdp2] losses=[13.63, 13.32, 13.48, 13.22, 13.22, 13.15]  finite=True  learned=True
        lora_has_grad=True  base_has_grad=False
RESULT: PASS — FSDP2 shards + trains a 4-bit QLoRA model (only LoRA learns)
```

i.e. FSDP2's parameter flattening/sharding + **CPU offload + bf16 mixed precision + activation
checkpointing** all accept the bnb-4-bit + LoRA params, and only the adapters learn — the part that
normally breaks. This is world_size=1, so it does not test cross-rank comms.

**What still needs 2 real GPUs (I have one, and it's an A2000 not a T4):** the full 2× run of
`fsdp2_qlora_sft.py` on Llama-3.1-8B and the 60-step loss-curve-equals-single-GPU check. Run it on a
2× NVIDIA box (Kaggle 2× T4, or a Windows machine with 2 GPUs via **WSL2** — native Windows can't do
NCCL FSDP2):

**Ready-to-run: [`task_b_kaggle_2xT4.ipynb`](task_b_kaggle_2xT4.ipynb)** — open in Kaggle, set the
accelerator to **GPU T4 ×2**, Run All. It installs deps, runs the single-GPU reference + the 2× T4
FSDP2 job, and plots the two loss curves (equivalence). Or from a shell on any 2× NVIDIA box:

```bash
# 2x GPU
accelerate launch --config_file unsloth_puzzles/fsdp2_config.yaml unsloth_puzzles/fsdp2_qlora_sft.py
# single-GPU reference loss for equivalence
CUDA_VISIBLE_DEVICES=0 python unsloth_puzzles/fsdp2_qlora_sft.py --single
```

Set `mixed_precision: fp16` in the YAML on Tesla T4 (Turing has no bf16).
