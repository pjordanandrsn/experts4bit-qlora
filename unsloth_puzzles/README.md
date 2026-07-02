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

**Confirmed on 2× T4 (Kaggle).** `fsdp2_qlora_sft.py` was run end-to-end on a Kaggle **2× T4** box via
[`run_kaggle.sh`](run_kaggle.sh): the single-GPU reference **and** the 2× T4 FSDP2 + QLoRA job both
train Llama-3.2-3B to completion, FSDP2 shards the bnb-4-bit base + LoRA across both cards (Gloo/NCCL
comms up, only the adapters learn), and the two converge to the same final loss. (Default is 3B so the
single-GPU reference fits one 16 GB T4; the recipe is identical for 8B on a ≥24 GB card or with FSDP
CPU-offload loading.)

**Equivalence, done honestly.** The two legs are compared with the **global batch held constant** —
`per_device_bs × grad_accum × world_size` — so the script divides `grad_accum` by `world_size` (4→2 on
2 GPUs) and both legs then consume the **same tokens per optimizer step**. Step-for-step loss is *not*
expected to be bit-identical under data-parallel FSDP2: the `DistributedSampler` reshards examples
across ranks, and the cross-rank gradient all-reduce sums in a different (non-deterministic) float
order than single-GPU accumulation. Equivalence is therefore shown by **(a)** identical tokens/step
across legs, **(b)** closely-tracking loss curves with no systematic drift, and **(c)** final
`train_loss` agreement within a few percent. `run_kaggle.sh` prints the `step / single / fsdp2` table
and `MAX_ABS_LOSS_DIFF`. (An initial run left `grad_accum` un-scaled, so the 2× leg trained a 2×-larger
global batch — its per-step losses sampled a different data schedule; the `//world_size` scaling above
is what makes the per-step comparison apples-to-apples.)

**Ready-to-run.** One Kaggle cell (2× T4, Internet on) — installs deps, runs both legs, prints the
equivalence table:

```bash
curl -sSL https://raw.githubusercontent.com/pjordanandrsn/experts4bit-qlora/triton-nf4/unsloth_puzzles/run_kaggle.sh | bash
```

or the notebook [`task_b_kaggle_2xT4.ipynb`](task_b_kaggle_2xT4.ipynb) (set the accelerator to **GPU
T4 ×2**, Run All). From a shell on any 2× NVIDIA box (or a Windows machine via **WSL2** — native
Windows can't do NCCL FSDP2):

```bash
# 2x GPU
accelerate launch --config_file unsloth_puzzles/fsdp2_config.yaml unsloth_puzzles/fsdp2_qlora_sft.py
# single-GPU reference loss for equivalence
CUDA_VISIBLE_DEVICES=0 python unsloth_puzzles/fsdp2_qlora_sft.py --single
```

Training is in **bf16** (native on Ampere+, emulated but functional on a T4), and the YAML uses
`mixed_precision: bf16` to match — so no fp16 `GradScaler` is involved. (fp16 would crash the
single-GPU leg: its `unscale_` kernel has no bf16 implementation and this bnb-4-bit + PEFT stack
produces a bf16 gradient there.)
