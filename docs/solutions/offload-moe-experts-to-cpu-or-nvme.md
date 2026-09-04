# How do I offload MoE experts to host RAM, or serve and train them from an NVMe arena?
<!-- summary: Bind pinned host RAM or a baked NVMe arena to a real model with the streaming loader and the nvme_residency, mxfp4_nvme_residency and nvme_train_residency engines. -->

**This is the model-level integration page**: the loaders and the `experts4bit-qlora` engines that bind pinned host RAM or an NVMe arena to a real model. Choosing a path by workload and by the memory tier that ran out is [`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md), the decision/router page; the arena bake, reader and tier primitives themselves are the kernel package's page, [stream-moe-experts-from-host-or-nvme.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/stream-moe-experts-from-host-or-nvme.md).

For host RAM, `load_moe_4bit_streaming(..., offload=True)` pins each layer's frozen 4-bit experts in CPU RAM and streams one layer to the GPU at a time. When the experts do not fit host RAM either, bake them into an arena with `grouped-nf4-gemm` and bind it with `enable_nvme_residency` or `enable_mxfp4_nvme_residency` (serving) or `enable_nvme_train_residency` (training): cold rows are read from NVMe on demand while a pinned-DRAM hot tier of `hot_rows` absorbs repeats.

## Symptoms

- "offload MoE experts to system RAM" / "experts fit on NVMe but not in CPU RAM" / "OOM-killed on host memory while loading".
- "serve DeepSeek-V4 from disk" — the expert store is larger than any rented pod's pinned host RAM.
- The pinned host copy of a 30B-class MoE is what runs out during QLoRA, or you want to train an adapter over experts that live on NVMe.

## Why it happens

Layer-granular offload bounds VRAM but not host RAM: its homes are the full `[E, ...]` packed stacks for every layer at once. Top-k routing touches a small fraction of the expert set per token, so the cold tail only has to be *reachable*, not resident. An arena is a baked, expert-row-addressable file; reads are `O_DIRECT` where the platform allows, so the page cache is neither needed nor helpful and host RAM does not have to exceed the arena ([`../RESIDENCY-ENGINES.md`](../RESIDENCY-ENGINES.md)).

## Which project solves it

**grouped-nf4-gemm** ([GitHub](https://github.com/pjordanandrsn/grouped-nf4-gemm), [PyPI](https://pypi.org/project/grouped-nf4-gemm/)) owns the host/NVMe primitives: the bake (`nvme_bake_nf4` re-quantises a bf16 or block-FP8 source to NF4; `nvme_arena.bake_expert_tensors` *relocates* native MXFP4 bytes verbatim), the reader, `nvme_residency.ColdTier`, and `capacity_for_bytes` for sizing `hot_rows` from measured free RAM. **experts4bit-qlora** owns binding the arena to a real model: `engines/nvme_experts.py` replaces the frozen module's forward for serving; `engines/nvme_train.py` leaves the adapter's forward alone and moves only the frozen base's *home* from pinned RAM to the arena. `[fast]` is the seam.

## Install

```bash
pip install "experts4bit-qlora[fast]"    # residency/NVMe/fast-kernel route: grouped-nf4-gemm's arena bake, reader, tier and kernels
pip install "experts4bit-qlora[train]"   # host-RAM training route: the streaming loader and pinned-host expert offload
```

## Smallest correct example

Needs: GPU + network + model download + a baked NF4 arena + local NVMe.

```python
import torch
from experts4bit_qlora import (enable_fast_train, enable_nvme_train_residency,
                               load_moe_4bit_streaming, verify_moe_4bit)

ARENA = "/nvme/qwen3-30b.nf4.arena"      # baked by grouped-nf4-gemm's nvme_bake_nf4
model, cfg = load_moe_4bit_streaming(
    "Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16, r=8, alpha=16,
    quant_type="nf4", arena=ARENA, arena_train=True,   # arena_train=True is REQUIRED for training
)
verify_moe_4bit(model, strict=True)
n_experts = getattr(cfg, "num_local_experts", None) or cfg.num_experts   # the loader's own rule
n = enable_nvme_train_residency(model, ARENA, hot_rows=n_experts)        # floor: at least num_experts
assert n > 0
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # required
n_fast = enable_fast_train(model, dgrad=True)
assert n_fast > 0
```

Serving a native-MXFP4 arena (DeepSeek-V4) is the other side of the seam — [`mxfp4-moe-training-and-residency.md`](mxfp4-moe-training-and-residency.md). Serving an NF4 arena uses `enable_nvme_residency(model, ARENA, hot_sets, hot_rows=...)` on a model loaded with `arena=ARENA` and *no* `arena_train`, with one `hot_sets` entry per MoE module from `hot_sets_from_profile`.

## Expected result

`enable_nvme_train_residency` returns the number of `ExpertsLoRA` modules moved onto the arena (one per MoE layer); `arena_train_stats(model)` reports the tier afterwards. Undersized `hot_rows` is refused at attach, not many steps in. `verify_moe_4bit(model, strict=True)` still passes because the experts remain `Experts4bit` stacks. On the serving side, `enable_nvme_residency` returns the number of modules patched and refuses to stamp a partial set.

## Supported scope

- Families in the register for this path: Qwen3-MoE and OLMoE (host-RAM ceiling receipts), DeepSeek-V4 Flash / Pro and gpt-oss (native MXFP4).
- Arena formats: NF4 quantize-at-bake (four segments per expert row) and native MXFP4 relocation (six segments, fused on read). The manifest's `bake_mode` records which provenance claim the arena supports.
- Environment: Linux, NVIDIA CUDA sm_80 or newer, grouped-nf4-gemm>=0.28.0 (Triton, Linux-only), local NVMe or a fast block device, pinned host RAM for the hot tier; CI tests Python 3.11.

## Limitations

- **No shipped tool bakes the training arena from a bf16 checkpoint** — open, claim `e4b.open.tr2-repro-gap`; `TRAIN_ARENA` in the trainer takes a *path* to a pre-baked arena.
- `hot_rows` floors are hard: at least `k` for decode, approaching `min(T*k, num_experts)` for a prefill batch, and at least `num_experts` for training. Undersizing raises rather than thrashing.
- Training on an arena lifts the **host RAM** ceiling, not VRAM: one layer's full `[E, ...]` stack is still device-resident, and gradient checkpointing is required.
- The serving engines refuse `ExpertsLoRA`-wrapped modules rather than discard the adapter; `enable_mxfp4_nvme_residency` and training are mutually exclusive on one load.
- The per-step cost of arena training does not travel across cards (an arena step includes an NVMe read); quote it with the card attached (notes on `e4b.offload.arena-vs-host-ram`).
- Bulk host-RAM offload at decode is transfer-bound; for speed use the pipelined engine ([`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md)).

## Use this page when…

- **you are wiring a model** — `load_moe_4bit_streaming(..., offload=True)`, the arena loader (`arena=`, `arena_train=True`), `enable_nvme_residency` / `enable_mxfp4_nvme_residency` / `enable_nvme_train_residency` — this page, the model-level integration page.
- **you are choosing** a path by workload and by the memory tier that ran out — [`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md), the decision/router page.
- **you are baking, reading or sizing the tier itself** — `nvme_bake_nf4`, `bake_expert_tensors`, the reader, `ColdTier`, `capacity_for_bytes` — the kernel package's [stream-moe-experts-from-host-or-nvme.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/solutions/stream-moe-experts-from-host-or-nvme.md).

## Related

- [`mxfp4-moe-training-and-residency.md`](mxfp4-moe-training-and-residency.md) — native MXFP4 arenas (gpt-oss, DeepSeek-V4).
- [`qlora-fused-moe-experts.md`](qlora-fused-moe-experts.md) — the training lane the arena feeds.
- [`../RESIDENCY-ENGINES.md`](../RESIDENCY-ENGINES.md) · [`../DEEPSEEK-V4.md`](../DEEPSEEK-V4.md) · [`../CHOOSING.md`](../CHOOSING.md) · [`../STATUS.md`](../STATUS.md)

## Evidence

Register: [`../claims.json`](../claims.json).

- `e4b.offload.arena-vs-host-ram` — measured: against a descending host-RAM cap the arena path completes where the pinned-RAM path is OOM-killed, on OLMoE, Gemma-4 and Qwen3-30B.
- `e4b.serve.deepseek-v4` — measured: DeepSeek-V4-Flash loads and generates with its experts served from an on-disk arena at small peak VRAM.
- `e4b.serve.informed-hot-sets` — measured: hot sets ranked from a routing profile beat index-ordered ones at identical VRAM on V4-Flash.
- `e4b.open.tr2-repro-gap` — open: no shipped arena bake for the training receipt.
