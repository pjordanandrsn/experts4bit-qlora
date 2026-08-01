# experts4bit-qlora

[![CI](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/experts4bit-qlora)](https://pypi.org/project/experts4bit-qlora/)

QLoRA fine-tuning and serving of **fused Mixture-of-Experts** weights on a single small GPU.

## The problem

transformers v5 stores MoE experts as one fused 3-D `nn.Parameter` per layer
(`OlmoeExperts`, `Qwen3MoeExperts`, …). bitsandbytes' 4-bit walker only replaces `nn.Linear`,
so it **silently skips the experts** — the overwhelming majority of a MoE's weights.
`load_in_4bit` "shrinks" the model while the experts stay in full precision
([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)).

`ExpertsNbit` quantizes exactly that fused stack, at selectable precision — `nf4`/`fp4`
(4-bit packed), `int8`/`fp8` (8-bit blockwise), `bf16`/`fp16` (passthrough) — with a
test-pinned fidelity ordering, so the precision knob is a measured trade rather than a vibe.
`Experts4bit` is its 4-bit face. Paired with a **streaming loader** and **per-expert LoRA**,
you can actually fine-tune and serve a real sparse MoE on reasonable hardware.

Runs on a **stock** `pip install bitsandbytes` — every feature has a reference path.

## What it buys you

*Measured on an RTX A2000 12 GB in a NAS's PCIe 3.0 x8 slot unless noted; see
[METHODOLOGY](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/METHODOLOGY.md) "Test host".*

- **It fits at all.** Full bf16 OLMoE-1B-7B (~13.9 GB) **OOMs** on a 12 GB card; in 4-bit it
  loads at **4.70 GB** and trains in <8 GB. The loader never materializes the bf16 model in
  CPU *or* GPU RAM (verified under a 3 GB container RAM cap).
- **It trains.** QLoRA on the frozen NF4 experts improves a held-out Alpaca eval from
  **1.4813 → 1.0290**.
- **It scales past VRAM.** `OFFLOAD_EXPERTS=1` streams frozen experts from pinned CPU RAM one
  layer at a time: **Qwen3-30B-A3B peaks at 7.16 GB, Gemma-4-26B-A4B at 8.47 GB** — both OOM
  without it.
- **It goes bigger than RAM.** Experts can stream from an on-disk arena instead: full
  **DeepSeek-V4-Flash (284B, 43L × 256E)** runs in **8.74 GiB** with 147 GB of experts on
  NVMe. See [DEEPSEEK-V4](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/DEEPSEEK-V4.md).
- **It serves what it trained.** Adapters run over the *exact* NF4 base they were trained
  against — no GGUF/AWQ re-quantization shifting the error surface.
- **It dials.** Spare VRAM converts to decode speed: keep K hot experts resident, stream the
  cold tail. Picking K **from a routing histogram rather than by index** bought **+57–120%**
  on gpt-oss-20b and **+37%** on DeepSeek-V4 at identical VRAM — index-ordered hot sets are
  statistically indistinguishable from pure streaming. See
  [RESIDENCY-ENGINES](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/RESIDENCY-ENGINES.md).
- **Faster where it counts — and be precise which comparison.** *4-bit vs bf16 on a GPU that
  already fits the model* is a **1.2–2.3× energy penalty**: NF4 is storage-only, the GEMM runs
  in bf16 either way, plus dequant. That is real and not hidden here. It inverts when memory
  binds. *Fused vs reference, both 4-bit and both offloaded — the path you actually train on*
  — is **1.75–1.81× faster per step** at **0.754×** peak VRAM and **0.797–0.846×** energy per
  step, across five datasets on a 30B-class MoE, held-out loss unchanged (worst Δ0.00723
  against a registered ≤0.05 band).

## Install

```bash
pip install experts4bit-qlora           # primitive + adapters (torch + bitsandbytes)
pip install "experts4bit-qlora[train]"  # + the streaming MoE trainer (transformers>=5.0)
pip install "experts4bit-qlora[fast]"   # + the fused grouped-GEMM path (grouped-nf4-gemm)
```

`e4b`, `experts4bit`, and `expertsnbit` are equivalent aliases.

## Quickstart

```python
import torch
from experts4bit_qlora import Experts4bit, ExpertsNbit, ExpertsLoRA

gate_up = torch.randn(8, 2 * 256, 128)   # [num_experts, 2*intermediate, hidden]
down    = torch.randn(8, 128, 256)       # [num_experts, hidden, intermediate]
base    = Experts4bit.from_float(gate_up, down, quant_type="nf4")
model   = ExpertsLoRA(base, r=8, alpha=16)          # only the adapters train

base8   = ExpertsNbit.from_float(gate_up, down, quant_type="int8")   # other precisions
```

For a **real** checkpoint use the streaming loader — it builds on `meta` and quantizes the
fused experts on the way to the GPU. Do **not** use stock `from_pretrained`: it leaves the
experts in full precision and OOMs.

```python
from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit

model, config = load_moe_4bit_streaming(
    "Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4")
verify_moe_4bit(model, strict=True)      # raises and lists any stack left in high precision
```

```bash
STEPS=150 R=8 OUT=./out python -m experts4bit_qlora.train      # fine-tune
ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer # serve it
MODEL=Qwen/Qwen3-30B-A3B python -m experts4bit_qlora.infer      # or just generate
```

> **OOM loading in 4-bit?** If you used `BitsAndBytesConfig(load_in_4bit=True)`, that path
> quantized only the `nn.Linear` layers and skipped the fused experts — they are still bf16.
> Switch to `load_moe_4bit_streaming`, then `verify_moe_4bit(model, strict=True)` to confirm.

## Which door?

| You want | Call |
|---|---|
| Train / maximum compatibility | nothing — the reference forward is the default |
| Faster frozen-expert **inference** on CUDA | `enable_fast(model)` (`[fast]`, 3.65× at bs=1) |
| Serve past VRAM — hot resident, cold **streamed** | `enable_pipelined_residency(model, hot_sets, k_slots=k)` |
| Hot resident, cold **computed on the host CPU** | `enable_cold_engine(model, hot_sets)` |
| Experts larger than host RAM — stream from **disk** | `enable_mxfp4_nvme_residency(model, arena, ...)` |
| Experts exceed VRAM, training or serving | `OFFLOAD_EXPERTS=1` / `load_moe_4bit_streaming(..., offload=True)` |

⚠️ The residency engines need **standalone** expert modules; `load_moe_4bit_streaming`
always wraps in `ExpertsLoRA`, which they refuse or skip. Details, the host-regime laws, and
how to build hot sets: [RESIDENCY-ENGINES](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/RESIDENCY-ENGINES.md).

## Supported models

The `ExpertsNbit` primitive and `ExpertsLoRA` adapters are **model-agnostic**. The streaming
loader supports these; unsupported architectures **fail fast with a clear error**.

| model | notes |
|---|---|
| **OLMoE** (1B-7B) | convergence-tested end-to-end; ~4.7 GB on a 12 GB card |
| **Qwen3-MoE / Qwen3.5-MoE** | same layout as OLMoE (verified byte-identical) |
| **Gemma-4** (text tower) | experts beside a parallel dense MLP, fused on disk |
| **GraniteMoe** | legacy `input_linear`/`output_linear` spellings, renamed on load |
| **gpt-oss** (20b/120b) | MXFP4 on disk + per-expert biases + clamped GLU; built bare |
| **DeepSeek-V4** (Flash/Pro) | MXFP4 experts + FP8 dense half; arena-servable; trainable — [details](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/DEEPSEEK-V4.md) |

## Training + expert offload

Training holds no dequantized-expert activations — the frozen base re-dequantizes from the
packed weights inside backward, so activation memory stays flat in the number of experts.

- **`QUANT_TYPE=nf4|fp4|int8|fp8|bf16|fp16`** picks the frozen base's storage end-to-end
  (loader → training → serving). Serve with the value you trained with; checkpoint metadata
  enforces it.
- **`OFFLOAD_EXPERTS=1`** keeps experts in pinned CPU RAM and streams one layer at a time.
  Peak GPU drops by roughly *(experts footprint − one layer)* at **+11% s/step**. Location
  changes, math does not — unit-tested including the gradient-checkpoint recompute path.
- **`enable_fast_train()`** is where the throughput is: same offload, plus the differentiable
  grouped kernel — **1.75–1.81× faster per step** at **0.754×** peak VRAM.

Diagnostics (default off): `E4B_OFFLOAD_STATS=1`, `E4B_OFFLOAD_ARENA=1`,
`E4B_EXPERT_PROFILE=out.jsonl`. See [OFFLOAD-TRANSFER-NOTES](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/OFFLOAD-TRANSFER-NOTES.md)
and [METHODOLOGY](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/METHODOLOGY.md) §11.

## Inference

`python -m experts4bit_qlora.infer` serves adapters over the same NF4 base they trained
against. Inference mode adds a decode fast-path, a fused 4-bit GEMV for single-row
projections (gated by a correctness probe, passing on stock bitsandbytes 0.49.x), and
prefetched expert offload — all `no_grad`-only; training paths untouched.

Decode grids, the shape-dependence analysis (prefetch and GEMV swing from +46% to −8%
depending on expert shape — measure, don't extrapolate), and kill-switches for A/B:
[INFERENCE](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/INFERENCE.md).

## Docs

| | |
|---|---|
| [STORAGE-MODES](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/STORAGE-MODES.md) | the storage-mode support matrix and what each mode promises |
| [RESIDENCY-ENGINES](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/RESIDENCY-ENGINES.md) | pipelined / cold / NVMe, hot-set selection, host-regime laws |
| [DEEPSEEK-V4](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/DEEPSEEK-V4.md) | the 284B worked example, arena bake, key mapping |
| [INFERENCE](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/INFERENCE.md) | decode grids, fast-path knobs |
| [SERVING](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/SERVING.md) | FastAPI shim + Docker |
| [METHODOLOGY](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/METHODOLOGY.md) | test hosts, protocols, the convergence result |
| [BENCHMARKS](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/BENCHMARKS.md) | benchmark scripts and how to run them |
| [BITSANDBYTES](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/docs/BITSANDBYTES.md) | relationship to bitsandbytes #1965, vendoring, credits |

## The package family

**`experts4bit-qlora`** owns everything *around* the expert GEMM: the fused-stack primitives
and per-expert LoRA, streaming loaders, offload, QLoRA training, serving, and residency.
**[`grouped-nf4-gemm`](https://pypi.org/project/grouped-nf4-gemm/)** owns the GEMM itself — a
single-launch grouped kernel decoding NF4 in-register with fp32 accumulation.
`pip install "experts4bit-qlora[fast]"` is the seam.

In one line: **the kernel makes one expert-stack matmul cheap; this package decides which
bytes are where** — quantized how, resident where, streamed when, trained with what adapters.

## Provenance

Every measured number traces to a committed script/test and a named host, with receipts under
[`bench/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.8.0/bench/) and [`docs/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.8.0/docs/), cited inline.
[`PROVENANCE.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.8.0/PROVENANCE.md) is the OpenTimestamps-anchored record for the **v0.2.0**
convergence result specifically; later additions are receipted in `bench/` and `docs/`.
Falsification work lives under [`audits/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.8.0/audits/).

## License

MIT. `experts4bit_qlora/_vendor/experts.py` is vendored from bitsandbytes (also MIT) pending
upstream merge.
