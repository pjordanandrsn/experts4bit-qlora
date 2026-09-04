# experts4bit-qlora

[![CI](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/experts4bit-qlora)](https://pypi.org/project/experts4bit-qlora/)

Train and serve **fused Mixture-of-Experts** models in 4-bit on hardware
that cannot hold them in bf16.

**The problem in one line:** `load_in_4bit=True` leaves a fused MoE's
expert weights in bf16, so the model still OOMs; this package quantises
exactly those experts, fine-tunes them with QLoRA, keeps them in host RAM
or on NVMe when they do not fit, and serves them on one consumer NVIDIA
GPU. **Canonical package:** `experts4bit-qlora` on PyPI (`import
experts4bit_qlora`); `e4b`, `e4b-qlora`, `experts4bit`, `expertsnbit` and
`experts-mxfp4` are lookup aliases. **Two repositories:** this one owns loading, quantisation
orchestration, adapters, training, residency and serving; the kernels it
calls through the `[fast]` extra live in
[`grouped-nf4-gemm`](https://github.com/pjordanandrsn/grouped-nf4-gemm).
**Environment:** Linux, a CUDA GPU, torch ≥ 2.2 and bitsandbytes ≥ 0.43
(the floors are `pyproject.toml`'s; Python 3.11 is what CI tests; the
kernels need Triton on an sm_80+ GPU). **The material limitation:** on a model that already fits in bf16,
4-bit here is a memory trade, not a speed-up, and on the measured
comparator it cost energy (`e4b.train.energy-honest.scoped-a2000`) — this
is for models that do not fit. Machine-readable capabilities and evidence:
[`docs/capabilities.json`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/capabilities.json)
and [`docs/claims.json`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/claims.json).

transformers v5 stores a MoE's experts as one fused 3-D parameter per
layer. bitsandbytes' 4-bit walker only replaces `nn.Linear`, so it
**silently skips the experts** — the overwhelming majority of the weights
([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)).
This package quantises exactly that fused stack (`Experts4bit`, the 4-bit
face of `ExpertsNbit`: nf4 / fp4 / int8 / fp8 / bf16 / fp16 storage, with
a test-pinned fidelity ordering), pairs it with a streaming loader and
per-expert LoRA so you can fine-tune, and serves the result through a
paged decode engine that is measured against each model's own attention.

**Current position, one page:** [`docs/STATUS.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/STATUS.md).
**Every number, with its evidence and status:** [`docs/claims.json`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/claims.json).

## Use this when

- `load_in_4bit=True` / `BitsAndBytesConfig` loads your MoE but the
  expert tensors (`gate_up_proj`, `down_proj`) stay bf16 and the model
  still OOMs — the experts are fused 3-D parameters, not `nn.Linear`.
- You need QLoRA or LoRA on the experts themselves, and PEFT or the
  bitsandbytes walker never sees them.
- The quantised experts fit in host RAM but not VRAM (stream per layer),
  or fit on NVMe but not host RAM (serve or train from an arena).
- You want to train a 30B-class MoE on a 12–24 GB consumer GPU (expert
  offload; `e4b.offload.fits-30b-class`), or serve one on an RTX 5090 —
  the only card the serving claims (`e4b.serve.tp.*`,
  `e4b.serve.buildout.*`) are measured on.
- You are choosing between the reference per-expert path, the batched
  path, the fused kernel path, host-streamed residency and the NVMe tier —
  [`docs/CHOOSING.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/CHOOSING.md) is the decision page.
- Your experts are NF4, native MXFP4 (gpt-oss, DeepSeek-V4), int4-b32 for
  serving, fp8 for the KV cache, or a mix across storage and residency
  tiers.

## Do not use this when

- The model is dense (no experts): bitsandbytes' own 4-bit path already
  covers every `nn.Linear`.
- The model already fits in bf16 with headroom: 4-bit is a memory trade
  there, and on the measured comparator it was slower and used more
  energy (`e4b.train.energy-honest.scoped-a2000` in the claims register —
  one card and one bitsandbytes development build, not a statement about
  every 4-bit path).
- You expect a general-purpose serving engine or a vLLM replacement: on
  the same box vLLM is ahead (`e4b.serve.h2h.vllm.same-box`); this is a
  measured 4-bit path for models that otherwise do not run at all.
- You need Windows, macOS, ROCm or a non-CUDA accelerator.
- The model family or expert layout is not in
  [`docs/ARCHITECTURE_SUPPORT.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/ARCHITECTURE_SUPPORT.md)
  — unsupported architectures fail fast with a named error; the
  accelerated paths fall back to the reference loop, so assert every
  `enable_*` count (a `0` looks identical to the per-expert loop).

## Start here

| | |
|---|---|
| [`docs/SOLUTIONS.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/SOLUTIONS.md) | one page per problem: symptoms, cause, install, smallest example, verification, limits |
| [`docs/capabilities.json`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/capabilities.json) | the machine-readable capability contract (entry points, environments, limitations, claim IDs) |
| [`docs/STATUS.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/STATUS.md) | the current position — claims tiered in the public register: confirmed, measured, measured-private, open, superseded, retired |
| [`docs/claims.json`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/claims.json) | every number with its evidence and status |
| [`docs/INDEX.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/INDEX.md) | what each document is and whether it is current |
| [`grouped-nf4-gemm`](https://github.com/pjordanandrsn/grouped-nf4-gemm) | the kernel package this one drives (`pip install "experts4bit-qlora[fast]"`) |
| [PyPI: experts4bit-qlora](https://pypi.org/project/experts4bit-qlora/) | the canonical distribution |
| [`llms.txt`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/llms.txt) · [`AGENTS.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/AGENTS.md) | orientation for language models and coding agents |
| The routing page for this project on cerinamroth.com (problem-first index, status, compatibility) | [https://cerinamroth.com/ml/experts4bit-qlora/](https://cerinamroth.com/ml/experts4bit-qlora/) |

## Install

```bash
pip install experts4bit-qlora           # primitive + adapters (torch + bitsandbytes)
pip install "experts4bit-qlora[train]"  # + the streaming MoE trainer
pip install "experts4bit-qlora[fast]"   # + the fused grouped-GEMM path (grouped-nf4-gemm)
```

`e4b`, `e4b-qlora`, `experts4bit`, `expertsnbit` and `experts-mxfp4` are
lookup aliases that install this package; always install and cite
`experts4bit-qlora`. Runs on stock bitsandbytes; every feature has a
reference path. Building from source — `pip install --no-build-isolation`,
or any build outside pip's isolated build environment — needs setuptools ≥ 77
for the PEP 639 license metadata in `pyproject.toml`; an ordinary
`pip install` gets it automatically through build isolation. The `[fast]`
extra's floor on `grouped-nf4-gemm` is `pyproject.toml`'s and is not
repeated here; which version of this package needs which kernel release,
and why, is the `compatibility` record in
[`docs/system-manifest.json`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/system-manifest.json),
validated in CI against `pyproject.toml`.

## Which door? Start from what does not fit

| what ran out | call | needs |
|---|---|---|
| nothing — just train a fused MoE | `load_moe_4bit_streaming(...)` | `[train]` |
| each step is slow | `enable_fast_train(model, dgrad=True)` | `[fast]` |
| …and `[fast]` will not build | `enable_batched_train(model)` | — |
| the experts do not fit VRAM | `load_moe_4bit_streaming(..., offload=True)` | — |
| the experts do not fit host RAM, serving | `enable_nvme_residency(...)` | `[fast]` + arena |
| …and they are native MXFP4 | `enable_mxfp4_nvme_residency(...)` | `[fast]` + arena |
| the experts do not fit host RAM, training | `enable_nvme_train_residency(...)` | `[fast]` + arena + grad ckpt |
| the dense side does not fit | `enable_dense_offload(model, "cuda")` | — |
| serving, want it faster | `enable_fast(model)` | `[fast]` |
| serving, spare VRAM to trade | `enable_pipelined_residency(model, hot_sets, k_slots=k)` | `[fast]` |

Reasoning and caveats for each: [`docs/CHOOSING.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/CHOOSING.md).
**Assert the return value** of every `enable_*`: `0` and "silently still
on the per-expert loop" look identical from the caller's side.

## Quickstart

```python
import torch
from experts4bit_qlora import Experts4bit, ExpertsLoRA, load_moe_4bit_streaming, verify_moe_4bit

# A real fused-MoE checkpoint, quantised on the way to the GPU (never bf16-resident):
model, config = load_moe_4bit_streaming(
    "Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
)
verify_moe_4bit(model, strict=True)   # raises if any expert stack is still high precision
```

```bash
STEPS=150 R=8 TRAIN_EXPERTS=1 OUT=./out python -m experts4bit_qlora.train      # QLoRA fine-tune
ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer                 # serve it
```

Do **not** load these models with stock `from_pretrained(...,
load_in_4bit=True)`: it quantises the `nn.Linear` layers, leaves the
experts in bf16, and OOMs.

## What is measured

Each row names its entries in `docs/claims.json`, which carry the value,
the conditions and the receipt path; the last column is the status there. **measured** means the receipt is in this repository;
**measured-private** means the run happened but the receipt lives in a
private audit tree and you cannot check it from here.

| | result | status |
|---|---|---|
| OLMoE-1B-7B fits a 12 GB card and trains (`e4b.train.olmoe-fits`, `e4b.train.olmoe-converges`) | 4.70 GB load; held-out eval 1.4813 → 1.0290 | measured |
| Expert offload trains 30B-class MoEs on 12 GB (`e4b.offload.fits-30b-class`) | Qwen3-30B-A3B peaks 7.16 GB, Gemma-4-26B-A4B 8.47 GB | measured |
| Fused training path, two 30B MoEs × five datasets (`e4b.train.flagship-matrix`) | 1.52–1.81× per step at 0.75–0.81× VRAM, loss parity, frozen stack bit-identical over 16.31 GB | measured |
| Arena vs pinned host RAM, at a descending cap (`e4b.offload.arena-vs-host-ram`) | 2.56× / 3.80× / 6.40× less host RAM (OLMoE / Gemma-4 / Qwen3-30B) | measured |
| Paged decode vs the model's own attention (`e4b.parity.*.paged-vs-own-attention`, `e4b.parity.gemma4.no-reference`, `e4b.parity.gemma4.fp8-share`) | indistinguishable on Granite (0.00229 nats), gpt-oss (0.00288) and Qwen3 (0.00173) against a chunk-free reference, each below its own floor; Gemma-4 has no reference at this resolution — its own cached forward swings −0.107 … +0.271 nats across windows — and the paged path's one measured cost there is the fp8 cache, 0.046 nats ([#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)) | measured-private |
| Per-family serving throughput on one rented RTX 5090 class (six families, same protocol; `e4b.serve.tp.*`) | Qwen3-30B 97 → 155 tok/s B=1 and 483 → 944 B=16; OLMoE 248 → 452; Granite 191 → 285; Mixtral 48 → 107; gpt-oss and Gemma-4 NF4 only (124, 71) — the refused arms are the build-out ([`SERVING-THROUGHPUT.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/SERVING-THROUGHPUT.md)) | measured |
| Single-stream Qwen3-30B-A3B on an RTX 5090 (`e4b.serve.b1.qwen3-30b.nf4.5090.2026-09`, `e4b.serve.b1.qwen3-30b.int4attn-calib.5090`) | ≈100 tok/s NF4; 204.6 tok/s with calibrated int4 attention + int4 experts | measured-private |
| Batched (B=16) Qwen3-30B-A3B on an RTX 5090 (`e4b.serve.b16.qwen3-30b.int4.5090`) | ≈1,238 tok/s aggregate | measured |
| Same box, same prompts, against vLLM (GPTQ-Int4) (`e4b.serve.h2h.vllm.same-box`) | vLLM ahead 1.47× at B=1, 1.55× at B=16 | measured-private |
| DeepSeek-V4-Flash (284B, 147 GB of experts on disk) (`e4b.serve.deepseek-v4`) | loads in ~10 s at 8.74 GiB peak VRAM and generates | measured |
| Informed hot sets vs by-index, identical VRAM (`e4b.serve.informed-hot-sets`) | +37.1% on DeepSeek-V4-Flash; the gain is a property of the host | measured |

Three things to read beside that table, because they change what it
means:

- **A parity delta is read against a per-model noise floor, never
  against zero.** Two arithmetically equivalent forwards of an MoE
  disagree, because rounding flips which experts the router picks; on
  gpt-oss 4.5% of layer-token choices flip and those tokens carry the
  whole disagreement. "Below the floor" means indistinguishable.
  [`docs/METHODOLOGY.md` §13.1](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/METHODOLOGY.md).
- **4-bit on a card that already fits the model was a 1.2–2.3× energy
  penalty on the measured comparator**, not a saving: one OLMoE-dims
  expert projection on an RTX A2000, dequantize-then-`linear` and a
  bitsandbytes 0.50-dev fork build's `matmul_4bit` routing against native
  bf16 (`e4b.train.energy-honest.scoped-a2000`). It inverts when memory
  binds. *Note, 2026-09-04:* the earlier wording "NF4 is storage-only and
  the GEMM runs in bf16 either way" was a universal mechanism statement
  and is withdrawn as such — bitsandbytes ≥ 0.50.0 can run supported
  ordinary 2-D 4-bit inference cells on the packed weights directly, while
  routed grouped MoE execution and training's input gradient are separate
  contracts ([`docs/BITSANDBYTES.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/BITSANDBYTES.md)).
  The measurement stands as its receipt made it.
- **Ratios travel; absolutes do not.** The 5090 class carries ~8.5%
  inter-box dispersion; the same config on two 4090s moved 8.6% in
  s/step. Quote the card, or quote a ratio.

## What was retired

Claims this project published and then withdrew, each with the
measurement that withdrew it, are listed in
[`docs/STATUS.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/STATUS.md#what-changed--retired-superseded-corrected)
and kept as `retired` entries in `docs/claims.json` so they stay
findable. The most recent: the "+0.047 ppl fp8 KV cost" on Qwen3 (below
the model's own floor), and the "+0.078 nats gpt-oss sinks/windows
defect" (the chunked oracle was the drifting arm, not the serving path).

## Scope

The primitives are model-agnostic. The streaming loader and trainer
handle SwiGLU fused-MoE families stored per-expert or pre-fused:
**OLMoE**, **Qwen3-MoE / Qwen3.5-MoE**, **Gemma-4** (text tower),
**GraniteMoe**, **gpt-oss** (MXFP4 experts with per-expert biases and a
clamped GLU, dequantised bit-identically), and **DeepSeek-V4** (Flash /
Pro). Which families load, run and CUDA-graph-capture, with the
evidence: [`docs/ARCHITECTURE_SUPPORT.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/ARCHITECTURE_SUPPORT.md).
Unsupported architectures fail fast with a clear error.

Known open: Gemma-4-26B-A4B's fp8 K cache wants finer groups on its
512-dim heads, and the family needs a parity instrument that survives
its batch-shape variance ([#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)); the model fails to load on
2 of 5 rented hosts
([#344](https://github.com/pjordanandrsn/experts4bit-qlora/issues/344));
no shipped tool bakes the training arena from a bf16 checkpoint yet.

## Docs

| | |
|---|---|
| [`docs/STATUS.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/STATUS.md) | what you get, what was retired, what is open — one page |
| [`docs/claims.json`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/claims.json) | every claim with value, hardware, status, evidence |
| [`docs/INDEX.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/INDEX.md) | what each of the 42 documents is, and whether it is current |
| [`docs/CHOOSING.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/CHOOSING.md) | which mode, and why |
| [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/METHODOLOGY.md) | hosts, protocols, every measurement's provenance |
| [`docs/SERVING-PARITY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/SERVING-PARITY.md) | paged decode vs each model's own attention |
| [`docs/SERVING-THROUGHPUT.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/SERVING-THROUGHPUT.md) | per-family decode throughput under one protocol, with the refusal list |
| [`docs/STORAGE-MODES.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/STORAGE-MODES.md) | the six storage modes and what each promises |
| [`docs/RESIDENCY-ENGINES.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/RESIDENCY-ENGINES.md) | residency engines, hot-set selection, host-regime laws |
| [`docs/SERVING.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/SERVING.md) | the HTTP shim and Docker deployment |
| [`docs/DEEPSEEK-V4.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/DEEPSEEK-V4.md) | V4's storage split, epilogue, arena bake |
| [`docs/BITSANDBYTES.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.35.0/docs/BITSANDBYTES.md) | relationship to bitsandbytes, prior art |

## The package family

- **`experts4bit-qlora`** (this repo) owns everything *around* the expert
  GEMM: the fused-stack primitives and per-expert LoRA, the streaming
  loaders, offload, training, the paged serving engine, hot-expert
  residency.
- **[`grouped-nf4-gemm`](https://pypi.org/project/grouped-nf4-gemm/)**
  owns the GEMM itself: one launch over 4-bit-packed expert stacks with
  in-register decode and fp32 accumulation, plus the fp8 paged decode
  attention and the decode glue kernels. `[fast]` is the seam.

The kernel makes one expert-stack matmul cheap; this package decides
which bytes are where.

## Provenance

Every number traces to a committed script and a named host, with
receipts under `bench/` and `docs/` — or, where the receipt is private,
the register says so. `PROVENANCE.md` is the OpenTimestamps-anchored
record for the v0.2.0 convergence result; anchored documents are never
edited in place (see `docs/INDEX.md`). Falsification work lives under
`audits/`.

## License

MIT ([`LICENSE`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/LICENSE)). `experts4bit_qlora/_vendor/experts.py`
is vendored from bitsandbytes (also MIT) pending upstream merge; its
notice is in [`THIRD_PARTY_NOTICES.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/THIRD_PARTY_NOTICES.md).
