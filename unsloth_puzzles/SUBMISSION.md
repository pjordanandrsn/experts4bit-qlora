# Unsloth Puzzles — submission

Two puzzles, both solved and **measured on the target hardware**. Developed on a single RTX A2000 12 GB;
the Tesla-T4 figures the rubric cares about were measured on Kaggle (1× T4 for Task A, 2× T4 for Task B).

## Start here — two runnable notebooks

| Task | Notebook | Runtime | Result (measured) |
|---|---|---|---|
| **A** — fused Triton NF4 kernel | [`task_a_triton_nf4.ipynb`](task_a_triton_nf4.ipynb) | GPU **T4** | **1.20× vs Unsloth `fast_dequantize`, 1.23× vs bnb** (geomean; 1.11–1.71× by shape) |
| **B** — FSDP2 + QLoRA on 2 GPUs | [`task_b_kaggle_2xT4.ipynb`](task_b_kaggle_2xT4.ipynb) | GPU **T4 ×2** | 1× vs 2× **MAX_ABS_LOSS_DIFF 0.028**, final loss 1.718 vs 1.724 (**0.35 %**) |

Each notebook is **self-contained** (installs deps, writes the kernel/script inline — no external repo
fetch) and ships with its **captured Kaggle outputs embedded**. Open in Kaggle, set the accelerator, Run
All to reproduce.

## Task A — [`../experts4bit_qlora/triton_nf4.py`](../experts4bit_qlora/triton_nf4.py)

A **single** Triton kernel does the full **double dequant** (bitsandbytes `compress_statistics=True`):
nested per-64-block `absmax` reconstructed on the fly, 16-entry NF4 codebook per nibble — one launch, no
intermediate buffer. Custom PTX (`bfe.u32`) nibble unpack; registered as a `torch.library.triton_op` so
it compiles `fullgraph=True` with no graph breaks. `your_dequantize_nf4(module)` matches the puzzle's
`test_dequantize` signature. Correctness is bit-exact vs `bnb.dequantize_4bit` (bf16) / within ULP (fp16).

## Task B — [`fsdp2_qlora_sft.py`](fsdp2_qlora_sft.py) + [`fsdp2_config.yaml`](fsdp2_config.yaml)

A transformers-native `SFTTrainer` under an FSDP2 accelerate config. `bnb_4bit_quant_storage=bf16` +
**no** `device_map` are what let FSDP2 flatten/shard the bnb-4-bit base alongside the LoRA params. FSDP2
parameter CPU offload, activation checkpointing, bf16 mixed precision, and transformer-layer auto-wrap
are all active. The global batch is held constant across launches (`grad_accum // world_size`) so the
single-GPU and 2× runs consume the same tokens per step, making the loss curves directly comparable.

## Reproduce & tests

- `pytest tests/` → **22 passed** (7 Triton incl. `torch.compile fullgraph`, plus the wider suite).
- Both notebooks Run-All on Kaggle; or use [`run_kaggle_triton.sh`](run_kaggle_triton.sh) (Task A, 1× T4)
  and [`run_kaggle.sh`](run_kaggle.sh) (Task B, 2× T4) as one-cell `curl | bash` runners.
- Rubric-by-rubric detail and honest caveats: [`README.md`](README.md).

## Honesty notes (what the numbers do and don't say)

- **Task A speedup is shape-dependent** — 1.71× vs Unsloth on small matrices, ~1.11–1.13× on large MLP
  matrices; the geomean (1.20×) clears the 1.15× bar, the big shapes individually sit just under it.
  Unsloth ≈ bnb on the T4 (not ~1.05× slower, as I'd first assumed — which is why I measured it).
- **Task B is not step-wise bit-identical** — data-parallel FSDP2 reshards examples across ranks and the
  all-reduce sums in a non-deterministic float order; equivalence = matched tokens/step + closely-tracking
  curves + final-loss agreement, not an "identical curve" claim.
- Building the Task A benchmark surfaced a real bug: `your_dequantize_nf4` called `float(qs.offset)` every
  invocation — a per-forward `cudaStreamSynchronize` on a CUDA tensor. Passing the offset as a device
  pointer (loaded in-kernel) removed it: the measured speedup *and* one fewer sync per training forward.
