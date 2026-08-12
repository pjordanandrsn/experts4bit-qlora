# experts4bit-qlora

[![CI](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/experts4bit-qlora)](https://pypi.org/project/experts4bit-qlora/)

QLoRA fine-tuning of **fused Mixture-of-Experts** weights on a single small GPU — the part that
doesn't fit anywhere else yet.

## The problem

transformers v5 stores MoE experts as one fused 3-D `nn.Parameter` per layer
(`OlmoeExperts`, `Qwen3MoeExperts`, …). bitsandbytes' 4-bit walker only replaces `nn.Linear`
modules, so it **silently skips the experts** — which are the overwhelming majority of a MoE's
weights. `load_in_4bit` "shrinks" the model but the experts stay in full precision
([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)).

`Experts4bit` is the primitive that 4-bit-quantizes exactly that fused stack. As of v0.2.0 it is
the 4-bit face of **`ExpertsNbit`**, which stores the same stack at selectable precision — `nf4`
/ `fp4` (4-bit packed), `int8` / `fp8` (8-bit blockwise), or `bf16` / `fp16` (passthrough) — with
a test-pinned fidelity ordering (`fp16` < `bf16` < `int8` < `fp8` < `nf4` < `fp4` reconstruction
error) so the precision knob is a measured trade, not a vibe. What each mode does and doesn't
promise is in [the support matrix](#storage-modes-the-support-matrix). This package pairs the
primitive with a **streaming loader** and **per-expert LoRA**, so you can actually *fine-tune* a
real sparse-MoE on reasonable hardware.

## What it buys you

> **Two hosts appear below and they are not interchangeable.** The fit/train/serve
> figures are an RTX A2000 12 GB in a NAS's PCIe 3.0 x8 slot (METHODOLOGY "Test
> host"); the fused-vs-reference cost ratios are RTX 4090s. Host matters more than
> it looks: the same config on two different 4090s moved **8.6 % in s/step** and
> **3.4× in idle power**, and only the within-pair *ratio* survived that.

- **It fits at all.** Full bf16 OLMoE-1B-7B is ~13.9 GB — it **OOMs** on a 12 GB card. In 4-bit
  it loads at **4.70 GB** and trains in <8 GB. The streaming loader never materializes the bf16
  model in CPU *or* GPU RAM (verified under a 3 GB container RAM cap).
- **It trains.** QLoRA on the frozen NF4 experts improves a held-out Alpaca eval from
  **1.4813 → 1.0290** (see [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/METHODOLOGY.md)).
- **It scales past VRAM (`OFFLOAD_EXPERTS=1`).** The frozen experts stream from pinned CPU RAM
  one layer at a time, so a fused-MoE whose 4-bit experts exceed the card can QLoRA-train on
  12 GB: **Qwen3-30B-A3B peaks at 7.16 GB, Gemma-4-26B-A4B at 8.47 GB** — both OOM *without*
  offload. Mechanics and cost under [Training + expert offload](#training--expert-offload).
- **It serves the fine-tune it made (`python -m experts4bit_qlora.infer`).** The adapters run
  over the *exact* NF4 base they were trained against — no GGUF/AWQ re-quantization shifting the
  error surface. OLMoE decodes at **1.44 tok/s in 1.68 GB** with prefetched offload (resident:
  3.08 tok/s at 4.86 GB); the same path decodes **Gemma-4-26B at 0.43 tok/s (6.2 GB)** and
  **Qwen3-30B-A3B at 0.22 tok/s (4.4 GB)** — models whose resident decode simply OOMs.
  *(v0 offload-path figures; the pipelined engine supersedes them for decode — see the dial below.)* See
  [Inference](#inference-serve-the-fine-tune-you-just-made).
- **It dials.** Spare VRAM converts to decode speed continuously — the pipelined engine keeps K
  hot experts/layer resident and streams the cold tail, and picking those K from a routing
  histogram (not by index) bought **+57–120%** decode at *identical* VRAM on gpt-oss-20b
  (receipts: [`bench/RESULTS-informed-hotsets.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/bench/RESULTS-informed-hotsets.md)).
  K=0 streams everything; K=all is fully resident; the middle is yours to trade.
- **It is faster and cooler — but be precise about which comparison.** *4-bit vs bf16 on a
  card that already fits the model* is a **1.2–2.3× energy penalty**: NF4 is storage-only, the
  GEMM runs in bf16 either way, plus dequant. That is real and not hidden here. It inverts when
  memory is the binding constraint — then it is the difference between running and not, and up
  to **4.4× lower energy/token** from the batch the freed memory unlocks. *Fused vs reference,
  both 4-bit and both offloaded* — the path you actually train on — is a straight win; see
  [Training + expert offload](#training--expert-offload).

## Install

```bash
pip install experts4bit-qlora           # primitive + adapters + benchmarks (torch + bitsandbytes)
pip install "experts4bit-qlora[train]"  # + the streaming MoE trainer (transformers>=5.0, datasets, ...)
pip install "experts4bit-qlora[fast]"   # + the fused grouped-GEMM inference path (grouped-nf4-gemm)
```

With `[fast]`, `enable_fast(model)` routes frozen-expert inference through
[grouped-nf4-gemm](https://pypi.org/project/grouped-nf4-gemm/)'s single-launch
fused kernel (NF4 decoded in-register inside the GEMM, fp32 accumulation) —
measured **3.65×** over the reference per-expert loop at bs=1 decode on OLMoE
geometry (A2000). Inference-only: training forwards fall back to the reference
recompute path automatically, and modules with custom activations or
non-nf4/64 storage are skipped rather than mis-activated.


### Which door? Start from what does not fit

Every mode exists because something ran out: VRAM, host RAM, or disk. Find your constraint,
not your model. Reasoning, caveats and requirements for each: **[docs/CHOOSING.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/CHOOSING.md)**.

| what ran out | call | needs |
|---|---|---|
| nothing — just train a fused-MoE | `load_moe_4bit_streaming(...)` | `[train]` |
| each step is slow | `enable_fast_train(model, dgrad=True)` | `[fast]` ≥ 0.7.0 |
| …and `[fast]` will not build | `enable_batched_train(model)` | — |
| the experts do not fit VRAM | `load_moe_4bit_streaming(..., offload=True)` | — |
| the experts do not fit host RAM, **serving** | `enable_nvme_residency(...)` | `[fast]` + arena |
| ...and they are native MXFP4 | `enable_mxfp4_nvme_residency(...)` | `[fast]` + arena |
| the experts do not fit host RAM, **training** | `enable_nvme_train_residency(...)` | `[fast]` + arena + grad ckpt |
| the **dense** side does not fit | `enable_dense_offload(model, "cuda")` | — |
| ...nor does it fit host RAM | `DenseDiskSource(path)` | — |
| serving, want it faster | `enable_fast(model)` | `[fast]` |
| serving, spare VRAM to trade | `enable_pipelined_residency(model, hot_sets, k_slots=k)` | `[fast]` |
| small GPU, strong CPU | `enable_cold_engine(model, hot_sets, dequant="auto")` | — |

```python
from experts4bit_qlora import enable_fast, enable_batched_train, enable_fast_train

n = enable_fast(model)           # inference; returns modules patched
n = enable_batched_train(model)  # training, no extras; assert n > 0
n = enable_fast_train(model)     # training via [fast]; assert n > 0 or you are on the loop
```

**Assert the return.** `0` and "silently still on the per-expert loop" look identical from
the caller's side, and the loop is what these exist to escape: ~10k sync-gated iterations
per forward at 256 experts over 40 layers.

#### Which training path?

Both replace that loop, and **the answer depends on scale** — a microbench and a real
model rank them oppositely. Measured at both, one training step each
([receipts](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/bench/dgrad-gate/RESULTS-dgrad-gate.md)):

| | A2000 microbench (hidden 512, E=256) | Qwen3-30B-A3B, 48 layers, A6000 |
|---|---|---|
| `enable_fast_train(model)` | 4.53x | 1.72x |
| `enable_fast_train(model, dgrad=True)` | ~23x | **2.52x — fastest** |
| `enable_batched_train(model)` | **24.01x** | 1.05x — no speedup, highest peak |

At toy width the per-expert Python loop is the cost, and batching anything wins. At real
width the matmuls dominate, and `enable_batched_train`'s whole-stack dequant — paid on
every forward *and* backward — stops being amortizable, while the `[fast]` kernels read
packed bytes and keep their edge. **Default to `enable_fast_train(model, dgrad=True)`.**
Reach for `enable_batched_train` when `grouped-nf4-gemm` will not build for your arch —
it needs no extras (pure torch plus the bitsandbytes this package already requires) — and
know that its whole-stack materialization also costs the most peak memory of any lane
(26.7 GB vs the reference's 23.1 GB on the 30B fixture).

`enable_batched_train` exists because [@jiwoon-ahn](https://github.com/jiwoon-ahn) proposed
the approach in [#38](https://github.com/pjordanandrsn/experts4bit-qlora/issues/38) — batch
the frozen expert projections with a single whole-stack dequant, sort token/expert pairs,
run the groups as `bmm`s — with measurements and a working implementation.

They are mutually exclusive and each refuses to patch over the other; call the matching
`disable_*` to switch.

`dgrad=True` (needs `grouped-nf4-gemm >= 0.7.0`) routes the `[fast]` lane's *backward*
through a single-launch dgrad kernel instead of its per-expert decode loop, which measured
78–84% of a training step. It materializes nothing. Requested against an older kernel
package it turns off with a warning rather than raising.

> **Numerics, measured at composed scale against fp32 truth.** Both paths sum experts in
> group-sorted rather than ascending-expert-id order. `dgrad=True` adds **no composed
> gradient error over the lane it extends** (4.97e-2 → 4.99e-2 mean vs the reference at
> 48 layers). More importantly, an fp32-compute truth arm over the same NF4 bytes shows
> **every lane sits on the composed bf16 noise floor** — ~3.4e-2 at 16 layers, ~5.2e-2 at
> 48, *including the reference loop itself* — with the `[fast]` lane landing closest to
> truth at 16 layers (2.95e-2 vs the reference's 3.41e-2). Divergence *between* lanes is
> two valid bf16 roundings drifting apart, not one lane being looser. Loss trajectories
> for every lane sit at ≤0.003 median |Δ|, far inside the 0.05 band the
> [fused-train gate](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/bench/fused-train-gate/RESULTS-fused-train-gate.md)
> registered.

**Picking the hot sets is the single largest lever**, and by-index is not a choice:
frequency-ranked sets beat index-ordered ones by **+37.1%** at identical VRAM on
DeepSeek-V4-Flash, where index-ordered was statistically indistinguishable from pure
streaming — 4.4 GiB spent for nothing, because an index-ordered set *is* a uniform random
draw. `expert_profile` builds the histogram, `hot_sets_from_profile` ranks it. The size of
the gain is a property of your **host** (+40% on a thin-link A2000, ~0% on a fat-PCIe L40S).
Engines, selection and the host-regime laws: **[docs/RESIDENCY-ENGINES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/RESIDENCY-ENGINES.md)**.

⚠️ Both host-RAM residency engines need **standalone** expert modules; `load_moe_4bit_streaming`
always wraps in `ExpertsLoRA`, which `enable_pipelined_residency` refuses outright.

Runs on a **stock** `pip install bitsandbytes` today — see "Relationship to bitsandbytes" below.
> **CPU-only hosts:** on first import bitsandbytes prints a "kernels"/backend
> notice — harmless, and not from this package.

`pip install e4b`, `pip install experts4bit`, and `pip install expertsnbit` are equivalent aliases of this package.

## Quickstart

```python
import torch
from experts4bit_qlora import Experts4bit, ExpertsNbit, ExpertsLoRA

# Freeze a fused expert stack in 4-bit, attach trainable per-expert LoRA.
gate_up = torch.randn(8, 2 * 256, 128)          # [num_experts, 2*intermediate, hidden]
down    = torch.randn(8, 128, 256)              # [num_experts, hidden, intermediate]
base    = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.float32)
model   = ExpertsLoRA(base, r=8, alpha=16)      # only the LoRA adapters train

# Same stack at other storage precisions (8-bit blockwise / 16-bit passthrough):
base8   = ExpertsNbit.from_float(gate_up, down, quant_type="int8", compute_dtype=torch.float32)
```

End-to-end OLMoE QLoRA fine-tune (needs a CUDA GPU + `[train]` extras):

```bash
STEPS=150 R=8 TRAIN_EXPERTS=1 TRAIN_ATTENTION=0 OUT=./out \
  python -m experts4bit_qlora.train
```

### Load a real model in 4-bit

The Quickstart above uses synthetic tensors. To quantize a **real** fused-MoE checkpoint, use the
streaming loader — it builds the model on `meta` and 4-bit-quantizes the fused experts on the way to
the GPU. Do **not** load these models with stock `from_pretrained`: bitsandbytes' 4-bit walker only
replaces `nn.Linear`, so it silently leaves the experts in full precision and OOMs (see
[The problem](#the-problem)).

```bash
# CLI — stream-load + generate (add ADAPTER=./out/adapter_best.pt to serve a fine-tune):
MODEL=Qwen/Qwen3-30B-A3B QUANT_TYPE=nf4 python -m experts4bit_qlora.infer
```

```python
import torch
from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit

model, config = load_moe_4bit_streaming(
    "Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
)
model.to("cuda")                      # skip when offload=True
verify_moe_4bit(model, strict=True)   # optional: assert the fused experts are actually 4-bit
```

`Qwen/Qwen3-30B-A3B` in `nf4` is ~20 GB resident — it **fits a 24 GB card** (e.g. L4/A5000) with no
offload, ~4–5 tok/s decode. On a ≤12 GB card add `OFFLOAD_EXPERTS=1` (`offload=True`), which streams
the frozen experts from pinned CPU RAM one layer at a time; sizes and grids are in the
[support matrix](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/support_matrix.md).

> **Troubleshooting — OOM loading in 4-bit?** If you used
> `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))`
> and ran out of memory, that path quantized only the `nn.Linear` layers and skipped the fused
> experts ([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849))
> — they are still in bf16. Switch to `load_moe_4bit_streaming` (above), then call
> `verify_moe_4bit(model, strict=True)`: it raises and lists any expert stack still left in high
> precision, so you can confirm the fix.

## Storage modes: the support matrix

Moved to **[docs/STORAGE-MODES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/STORAGE-MODES.md)** — The full storage-mode support matrix (ExpertsNbit vs Experts4bit, compatibility, known limitations, the headline-number reading, reproduction + validation grids).

## Training + expert offload

Training holds no dequantized-expert activations: the frozen base projections re-dequantize from
the packed weights inside backward (`ExpertsNbit._project`), so activation memory stays flat in
the number of experts — on any released bitsandbytes, for every storage scheme. Two knobs:

- **`QUANT_TYPE=nf4|fp4|int8|fp8|bf16|fp16`** selects the frozen base's storage precision
  end-to-end (loader → training → serving). Default `nf4`; serve with the same value you trained
  with (the checkpoint metadata now enforces this). Aliases `bfloat16`/`float16` accepted;
  anything else fails before any checkpoint I/O — see
  [the support matrix](#storage-modes-the-support-matrix).
- **`OFFLOAD_EXPERTS=1`** keeps the frozen experts in pinned CPU RAM (set `OFFLOAD_PIN=0` to skip
  pinning) and streams one layer to the GPU at a time — GPU-resident only for that layer's
  forward and its gradient-checkpoint recompute, evicted after. Peak GPU drops by roughly
  *(experts footprint − one layer)* at the cost of one PCIe transfer per layer per pass
  (**+11 % s/step** on the OLMoE A/B). *Offload on its own* is a capacity feature — it changes
  *what fits*, not how fast. Offloading changes tensor location, not math — unit-test-verified,
  including the gradient-checkpoint recompute path. Offloaded *training* requires gradient
  checkpointing (the shipped trainer always enables it); the unsupported non-checkpointed
  combination fails loudly rather than mis-training. Details in
  [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/METHODOLOGY.md) §11.
- **`enable_fast_train()` makes the offload path faster, not just smaller** — and this is
  where the throughput is. Both arms run `offload=True` + gradient checkpointing; the only
  difference is whether the differentiable grouped kernel is on. Across **two** 30B-class MoEs
  (Qwen3-30B-A3B and Gemma-4-26B-A4B), five datasets each, 200 steps per cell:
  **1.52–1.81× faster per step** at **0.75–0.81×** peak VRAM and **0.86–0.92×** energy per step,
  with loss parity holding on both registered criteria (`|Δ final-**train**-loss| ≤ 0.05` and
  median step-wise `|Δ| ≤ 0.05`; worst cell 0.03653). The fused path *reproduces* the reference
  rather than trading accuracy for speed. The second model reproduces the *direction* of every
  cost result at lower magnitude, which is what its protocol registered — not a numeric match.
  Receipts: [`bench/flagship-matrix/`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/flagship-matrix/RESULTS-flagship-matrix.md)
  and [`bench/fused-train-gate/`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/fused-train-gate/RESULTS-fused-train-gate.md)
  (48-layer gate: 1.65×, 0.768× VRAM, frozen 4-bit stack verified bit-identical over
  **16.31 GB hashed**, with a byte-flip positive control that fires). Single same-process A/B
  pairs per cell, so the ratios are indicative, not variance-bounded.

Transfer diagnostics (default off): `E4B_OFFLOAD_STATS=1` prints per-layer H2D bandwidth, prefetch
stall/slack, and a one-shot PCIe-link + ceiling report; `E4B_OFFLOAD_ARENA=1` consolidates each
layer's four expert tensors into two per-dtype copies. What they measured on the reference host —
and why offload is PCIe-bound there — is in
[`docs/OFFLOAD-TRANSFER-NOTES.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/OFFLOAD-TRANSFER-NOTES.md).

## Scope

The `ExpertsNbit` primitive and `ExpertsLoRA` adapters are **model-agnostic**. The **streaming
loader / trainer** (`python -m experts4bit_qlora.train`) supports SwiGLU fused-MoE architectures —
experts stored either **per-expert** or already-**fused** on disk:

- **OLMoE** (OLMoE-1B-7B) — convergence-tested end-to-end; fits a 12 GB card at ~4.7 GB.
- **Qwen3-MoE / Qwen3.5-MoE** — same checkpoint + module layout as OLMoE (verified
  byte-identical); structurally tested.
- **Gemma-4 (text tower)** — different internally (experts at `layers.{i}.experts` beside a
  parallel dense MLP + a custom router; experts fused on disk) — handled and structurally tested.
- **GraniteMoe** (Granite-3.0-1b-a400m / 3b-a800m, PowerMoE-3b) — experts at
  `layers.{i}.block_sparse_moe.experts`, fused on disk under the legacy
  `input_linear`/`output_linear` spellings (the loader applies the same renames transformers'
  own converter does); handled and structurally tested. The 1b/3b checkpoints fit a 12 GB card
  without offload.
- **gpt-oss** (gpt-oss-20b / 120b) — experts shipped as MXFP4 blocks/scales with per-expert
  biases and a clamped-GLU epilogue; the loader dequantizes the exact released bytes
  (bit-identical) and builds a faithful NF4 expert (`GptOssExperts4bit`, built bare — the
  generic `ExpertsLoRA` assumes standard SwiGLU). Loads, offloads, and serves through
  hot-expert residency; run end-to-end on real 20b weights
  (`bench/RESULTS-gptoss-hybrid-ab.md`).

- **DeepSeek-V4 (Flash / Pro)** — per-expert **MXFP4** experts with a clamped-SwiGLU
  epilogue, and a **block-scaled FP8** dense half served by `fp8_blocks` at ~1 byte/param.
  Full V4-Flash (43 layers x 256 experts, 284B) loads in ~10 s at **8.74 GiB peak VRAM** and
  generates, with 147 GB of experts served from an on-disk arena. Trainable.
  See [docs/DEEPSEEK-V4.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/DEEPSEEK-V4.md).

The SwiGLU four are covered by `tests/test_loader_architectures.py`; gpt-oss by
`tests/test_hot_residency_gptoss.py` and the bench receipts. Real Qwen3/Gemma weights (26–35B)
need a ≥24 GB card — or the expert-offload path above — to fit 12 GB. Unsupported architectures
**fail fast with a clear error**; PRs for more welcome.

## Inference: serve the fine-tune you just made

The adapters were trained against *this exact* NF4 base (same codebook, same per-expert absmax).
`python -m experts4bit_qlora.infer` serves them over that same base — no re-quantization to
GGUF/AWQ, so the quantization error at serving time is identical to what training saw:

```bash
ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer            # generate
OFFLOAD_EXPERTS=1 BENCH_TOKENS=128 python -m experts4bit_qlora.infer       # timed decode bench
```

Inference mode adds a single-token decode fast-path, a fused 4-bit GEMV, and prefetched
expert offload (layer `L+1` copies on a side stream while `L` computes). Mechanics, the
kill-switches, and the shape-dependence analysis: **[docs/INFERENCE.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/INFERENCE.md)**.

| model | config | tok/s | peak GPU |
|---|---|:---:|:---:|
| OLMoE-1B-7B | resident (experts on GPU) | 3.08 | 4.86 GB |
| OLMoE-1B-7B | offload, serial | 0.40 | 1.45 GB |
| OLMoE-1B-7B | **offload + prefetch** | **1.44** | **1.68 GB** |
| Gemma-4-26B-A4B | resident | OOM | — |
| Gemma-4-26B-A4B | **offload + prefetch** | **0.43** | **6.16 GB** |
| Qwen3-30B-A3B | resident | OOM | — |
| Qwen3-30B-A3B | **offload + prefetch** | **0.22** | **4.41 GB** |

Capability, not throughput — and **the levers are shape-dependent**. Prefetch is the whole
result at OLMoE scale (3.65× over serial) while GEMV is neutral; at 26–30B scale GEMV swings
from **+46% on Gemma-4** to **−8% on Qwen3-30B**. Measure your model; do not extrapolate
across shapes. *(v0 offload-path figures — the pipelined engine supersedes them for decode.)*

### CUDA-graph decode capture

Decode issues ~89 kernel launches per layer-step; on OLMoE/3090 that was **288.8 ms of 562.4 ms**
of host time across 45,501 launches. `capture_decode` replays the whole step as one graph:

```python
from experts4bit_qlora import capture_decode, probe_capture

report = probe_capture(model, input_ids)          # does this model support it? measured, not assumed
dec, first = capture_decode(model, input_ids, max_length=len(prompt) + 128)
tok = first
for _ in range(127):
    tok = dec.step(tok)
dec.reset(input_ids)                              # required between generations
```

**What it is worth, measured** (16 new tokens, greedy):

| | speedup |
|---|---|
| 2-layer fixtures (qwen2_moe, qwen3_moe, granitemoe, hunyuan_v1_moe, glm4_moe, dots1, olmoe) | 4.4–5.6× |
| OLMoE-1B-7B-0924-Instruct, 3090 | **1.11×** |
| Qwen3-30B-A3B, A5000 | **1.04×** |

The gain is inversely proportional to real GPU work per step — which is what a *fixed* per-step
launch cost predicts. Take it for small models and for the zero-host-sync contract it enforces
(capture throws on a sync inside the region, so a successful capture proves the contract holds).
Do not expect it to matter at 30B.

Two costs stated plainly. `StaticCache` is allocated to `max_length` **up front**, so a large
`max_length` buys the fixed shape with memory. And `step()` is greedy argmax with no logits
processors, no stopping criteria and no streamer — which is exactly why the HTTP server does
**not** use it: `_generate_once` needs sampling, repetition penalty, stop signals and streaming,
and reimplementing those on a captured step to gain ~4% would be a bad trade against the risk.

`probe_capture` reports support rather than assuming it, and tells a bf16 argmax tie from a real
defect by teacher-forcing the same tokens down both paths and comparing logits. Both real-weight
models above replay **bit-identical** to eager. `qwen3_next` is not capturable — `StaticCache`
does not cover LinearAttention layers.

## Serving over HTTP (Docker)

Moved to **[docs/SERVING.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/SERVING.md)** — The FastAPI serving shim + Docker deployment (endpoints, env knobs, the localhost-by-default / E4B_TOKEN posture).

## Benchmarks

Moved to **[docs/BENCHMARKS.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/BENCHMARKS.md)** — The benchmark scripts and how to run them (memory wall, tokens-per-joule, the upstream matmul_4bit comparison).

## Docs

| | |
|---|---|
| [CHOOSING.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/CHOOSING.md) | which mode to use, and why — the long form of the table above |
| [METHODOLOGY.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/METHODOLOGY.md) | hosts, protocols, every measurement's provenance |
| [STORAGE-MODES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/STORAGE-MODES.md) | the nf4/fp4/int8/fp8/bf16/fp16 support matrix |
| [RESIDENCY-ENGINES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/RESIDENCY-ENGINES.md) | the four engines, hot-set selection, host-regime laws |
| [INFERENCE.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/INFERENCE.md) | decode fast-paths, grids, shape-dependence |
| [DEEPSEEK-V4.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/DEEPSEEK-V4.md) | V4's storage split, epilogue, arena bake, key mapping |
| [SERVING.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/SERVING.md) | the FastAPI shim and Docker deployment |
| [BENCHMARKS.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/BENCHMARKS.md) | the benchmark scripts and how to run them |
| [BITSANDBYTES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/BITSANDBYTES.md) | relationship to bitsandbytes, prior art |
- **[docs/ARCHITECTURE_SUPPORT.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/ARCHITECTURE_SUPPORT.md)** — which architectures load, run and capture, with the evidence bundle and the fixture-vs-real-checkpoint caveat.

## The package family — how the pieces fit

Two packages, one seam:

- **`experts4bit-qlora`** (this repo; aliases `e4b`, `experts4bit`, `expertsnbit`)
  owns everything *around* the expert GEMM: the fused-stack 4-bit primitives and
  per-expert LoRA, the streaming loaders (five architectures above), expert
  offload, QLoRA training, HTTP serving, and hot-expert residency. It runs
  complete on stock bitsandbytes — every feature has a reference path.
- **[`grouped-nf4-gemm`](https://pypi.org/project/grouped-nf4-gemm/)** owns the
  expert GEMM itself: a single-launch grouped kernel that decodes NF4
  in-register inside the mainloop with fp32 accumulation, replacing the
  dequant-then-GEMM round trip. `pip install "experts4bit-qlora[fast]"` is the
  seam — `enable_fast()` routes frozen-expert inference through it (3.65× at
  bs=1 decode on the dev card), and `enable_hot_residency()` runs its hot and
  cold stacks on the same kernel. The kernel repo carries its own registered
  claims and receipts (fidelity ordering, energy-per-token, 26→170 SM
  robustness).

Division of labor in one line: **the kernel makes one expert-stack matmul
cheap; this package decides which bytes are where** (quantized how, resident
where, streamed when, trained with what adapters).

## Relationship to bitsandbytes

Moved to **[docs/BITSANDBYTES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/docs/BITSANDBYTES.md)** — How ExpertsNbit/Experts4bit relate to bitsandbytes #1965, the vendored-copy shim, and the prior-art credits.

## Provenance & audits

Every measured number above traces to a committed script/test and a named host, with receipts
under [`bench/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.16.3/bench) and
[`docs/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.16.3/docs) — cited inline at
each claim. **Scope note (2026-07-28):** [`PROVENANCE.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/PROVENANCE.md)
is the OpenTimestamps-anchored record for the **v0.2.0** convergence result specifically; the
0.5.0–0.6.3 additions (fused kernel, hot-set residency, gpt-oss, storage modes) are receipted in
`bench/` and `docs/`, not in that file. It is OpenTimestamps-anchored: `ots verify
PROVENANCE.md.ots PROVENANCE.md` checks the on-disk bytes against the calendar proof, the footer
carries the hash-chain of prior revisions, and superseded proofs are retained in
[`.ots-history/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.16.3/.ots-history/). Falsification work lives under [`audits/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.16.3/audits/) — most
recently the audit of unsloth-zoo's MoE-4bit fix that produced unsloth-zoo#849/#850
([`audits/unsloth-zoo-4032/REPORT.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/audits/unsloth-zoo-4032/REPORT.md)).

## License

MIT (see [LICENSE](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.16.3/LICENSE)). `experts4bit_qlora/_vendor/experts.py` is vendored from
bitsandbytes (also MIT) pending upstream merge.
