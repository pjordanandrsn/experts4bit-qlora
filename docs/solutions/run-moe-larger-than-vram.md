# How can I run a Mixture-of-Experts model larger than my GPU's VRAM?
<!-- summary: Choose a residency path by workload and by the memory tier that ran out: pinned-host expert streaming, dense offload, profiled hot residency, or an NVMe arena. -->

Start from what ran out. If the 4-bit experts exceed VRAM, `load_moe_4bit_streaming(..., offload=True)` homes them in pinned host RAM and streams one layer at a time; if the dense side does not fit, `enable_dense_offload`; if the experts do not fit host RAM either, an NVMe arena; if you have spare VRAM to trade at serve time, `enable_pipelined_residency` with hot sets chosen from a routing profile.

## Symptoms

- "OOM even after 4-bit" on a 30B-class MoE (Qwen3-30B-A3B, Gemma-4-26B-A4B) on a 12 GB or 24 GB card.
- "run a MoE larger than VRAM" / "the experts don't fit" / "stream experts from host RAM".
- Unsure which path to take: the reference per-expert loop, the batched trainer, the fused kernel, host-streamed experts, or NVMe.
- The card fits the experts but not the attention and embeddings (the dense side).

## Why it happens

A MoE's weights are mostly experts, and each token touches only its top-k of them, so the whole model never has to be resident: the dense side plus one layer's experts is enough for a forward, and at decode only the routed rows are needed. What ran out decides where the rest lives — pinned host RAM across PCIe, or an on-disk arena read at the device link — and frozen 4-bit storage keeps the bytes that move small.

## Which project solves it

**experts4bit-qlora** decides which expert bytes are where: the streaming loader with `offload=True`, `enable_dense_offload` for the non-expert weights, `hot_sets_from_profile` for choosing resident experts, `enable_pipelined_residency` for the serve-time hot/cold split, and the `enable_nvme_*` engines. **grouped-nf4-gemm** ([GitHub](https://github.com/pjordanandrsn/grouped-nf4-gemm), [PyPI](https://pypi.org/project/grouped-nf4-gemm/)) supplies the fused grouped GEMM the pipelined engine runs on and the arena reader and tier the NVMe engines bind to; `[fast]` is the seam.

| what ran out | call | needs |
|---|---|---|
| the experts do not fit VRAM | `load_moe_4bit_streaming(..., offload=True)` (`OFFLOAD_EXPERTS=1` in the CLIs) | `[train]` |
| the dense side does not fit | `enable_dense_offload(model, "cuda")`; `DenseDiskSource(path)` when host RAM cannot hold it either | — |
| the experts do not fit host RAM, serving | `enable_nvme_residency(...)` / `enable_mxfp4_nvme_residency(...)` | `[fast]` + arena |
| the experts do not fit host RAM, training | `enable_nvme_train_residency(...)` | `[fast]` + arena + grad ckpt |
| serving, spare VRAM to trade | `enable_pipelined_residency(model, hot_sets, k_slots=k)` | `[fast]` |
| small GPU, strong CPU | `enable_cold_engine(model, hot_sets, dequant="auto")` | — |

## Install

```bash
pip install "experts4bit-qlora[train]"   # host-RAM training route: loader + pinned-host expert streaming; no kernel package needed
pip install "experts4bit-qlora[fast]"    # residency/NVMe/fast-kernel route: + grouped-nf4-gemm for the residency engines and NVMe arenas
```

## Smallest correct example

Needs: GPU + network + model download.

```python
import torch
from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit

model, config = load_moe_4bit_streaming(
    "Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
    offload=True, pin=True, prefetch=True,   # prefetch: next layer's H2D copy overlaps this layer (no_grad only)
)
verify_moe_4bit(model, strict=True)
model.eval()
# do NOT call model.to("cuda"): the experts live in pinned host RAM by design
```

Or from the CLI: `OFFLOAD_EXPERTS=1 BENCH_TOKENS=128 python -m experts4bit_qlora.infer`. Dense side too large too? `enable_dense_offload(model, "cuda")` composes with it.

## Expected result

`verify_moe_4bit(model, strict=True)` returns without raising, and `torch.cuda.max_memory_allocated()` during a forward stays near one layer's experts plus the dense side rather than the whole model. `enable_dense_offload` returns a non-empty list of per-layer handles, described by `dense_offload_report(handles)`. Every `enable_*` returns a count or a non-empty handle list, or raises — assert it.

## Supported scope

- Families: those of the loader ([`bitsandbytes-moe-load-in-4bit-still-ooms.md`](bitsandbytes-moe-load-in-4bit-still-ooms.md)). Offload identity is tested on nf4 / int8 / bf16; fp4 / fp8 / fp16 ride the same code path ([`../STORAGE-MODES.md`](../STORAGE-MODES.md)).
- Offload training requires gradient checkpointing (`use_reentrant=False`); the shipped trainer always enables it.
- Environment: Linux, NVIDIA CUDA, torch>=2.2, bitsandbytes>=0.43, transformers>=5.0; CI tests Python 3.11. Residency engines need grouped-nf4-gemm>=0.28.0 (Triton, Linux-only, sm_80 or newer).

## Limitations

- Bulk layer-granular offload is PCIe-bound for decode at 26–30B scale; the v0 decode grid in [`../INFERENCE.md`](../INFERENCE.md) is **superseded** for decode by the pipelined and paged engines (claim `e4b.retired.inference-md-decode-grid`).
- Pick hot sets from a routing histogram (`E4B_EXPERT_PROFILE`, `hot_sets_from_profile`), never by index: an index-ordered set is a uniform random draw. The gain is a property of the host link and did not replicate on a fat-PCIe box ([`../RESIDENCY-ENGINES.md`](../RESIDENCY-ENGINES.md)).
- `enable_pipelined_residency` accepts the `ExpertsLoRA` wrapper the loader installs and patches its base. The patch runs only while the wrapper delegates to the base (eval mode, `no_grad`, an adapter that provably contributes nothing — an untrained `B`); with a trained adapter it installs, never runs, and warns, which is why `experts4bit_qlora.serve` treats residency and trained adapters as mutually exclusive. Assert the count and check the served path ([`../CHOOSING.md`](../CHOOSING.md), [`../RESIDENCY-ENGINES.md`](../RESIDENCY-ENGINES.md)).
- `enable_hot_residency` is deprecated in favour of `enable_pipelined_residency`.
- Absolutes are host-specific; only ratios travel (claim `e4b.host.ratios-travel-absolutes-do-not`).

## Related

- [`offload-moe-experts-to-cpu-or-nvme.md`](offload-moe-experts-to-cpu-or-nvme.md) — the host-RAM and NVMe arena details.
- [`qlora-fused-moe-experts.md`](qlora-fused-moe-experts.md) — training on the offloaded experts.
- [`serve-large-moe-on-a-consumer-gpu.md`](serve-large-moe-on-a-consumer-gpu.md) — the paged serving engine.
- [`../CHOOSING.md`](../CHOOSING.md) · [`../RESIDENCY-ENGINES.md`](../RESIDENCY-ENGINES.md) · [`../STATUS.md`](../STATUS.md)

## Evidence

Register: [`../claims.json`](../claims.json).

- `e4b.offload.fits-30b-class` — measured: with `OFFLOAD_EXPERTS=1`, Qwen3-30B-A3B and Gemma-4-26B-A4B train on 12 GB where they OOM without it.
- `e4b.serve.informed-hot-sets` — measured: profile-ranked hot sets beat by-index at identical VRAM; the gain is a property of the host.
- `e4b.serve.deepseek-v4` — measured: DeepSeek-V4-Flash loads and generates with its experts served from an on-disk arena.
- `e4b.host.ratios-travel-absolutes-do-not` — measured: absolute s/step and tok/s are host-specific.
- `e4b.retired.inference-md-decode-grid` — superseded: the v0 offload decode grid is not the number to quote.
