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

## What it buys you (measured on an RTX A2000 12 GB — in a NAS's PCIe 3.0 x8 slot; see METHODOLOGY "Test host")

- **It fits at all.** Full bf16 OLMoE-1B-7B is ~13.9 GB — it **OOMs** on a 12 GB card. In 4-bit
  it loads at **4.70 GB** and trains in <8 GB. The streaming loader never materializes the bf16
  model in CPU *or* GPU RAM (verified under a 3 GB container RAM cap).
- **It trains.** QLoRA on the frozen NF4 experts improves a held-out Alpaca eval from
  **1.4813 → 1.0290** (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)).
- **It scales past VRAM (`OFFLOAD_EXPERTS=1`).** The frozen experts stream from pinned CPU RAM
  one layer at a time, so a fused-MoE whose 4-bit experts exceed the card can QLoRA-train on
  12 GB: **Qwen3-30B-A3B peaks at 7.16 GB, Gemma-4-26B-A4B at 8.47 GB** — both OOM *without*
  offload. Mechanics and cost under [Training + expert offload](#training--expert-offload).
- **It serves the fine-tune it made (`python -m experts4bit_qlora.infer`).** The adapters run
  over the *exact* NF4 base they were trained against — no GGUF/AWQ re-quantization shifting the
  error surface. OLMoE decodes at **1.44 tok/s in 1.68 GB** with prefetched offload (resident:
  3.08 tok/s at 4.86 GB); the same path decodes **Gemma-4-26B at 0.43 tok/s (6.2 GB)** and
  **Qwen3-30B-A3B at 0.22 tok/s (4.4 GB)** — models whose resident decode simply OOMs. See
  [Inference](#inference-serve-the-fine-tune-you-just-made).
- **Honest caveat — this is a memory technology, not an energy one.** On a GPU that *already*
  fits the model, 4-bit is a **1.2–2.3× energy penalty** (NF4 is storage-only; the GEMM runs in
  bf16 either way, plus dequant). The energy win only shows up when memory is the binding
  constraint — then it's the difference between running and not, and up to **4.4× lower
  energy/token** from the batch that freed memory unlocks. Numbers and method in the docs.

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

```python
from experts4bit_qlora import enable_fast, disable_fast
enable_fast(model)    # returns the number of expert modules patched
```

**Hot-expert residency** (`enable_hot_residency`, needs `[fast]` — it runs on
the fused kernel and raises at enable time with an install hint when the
kernel is missing) is the constrained-card path: it pins each MoE layer's
*hottest* experts in VRAM (fused kernel, zero transfer) and streams only the
cold tail from pinned host RAM per token — finer-grained than the whole-layer
residency GGUF runtimes place at. gpt-oss experts (clamped-GLU epilogue +
per-expert biases) are supported alongside the standard SwiGLU architectures.

**Pick the hot sets from a routing histogram, not by index** — measured
2026-07-20 (`bench/RESULTS-informed-hotsets.md`), the decode gain tracks
routing coverage on every architecture tried: gpt-oss-20b K=4 informed
**+56%** / K=8 **+120%** over the all-cold floor (naive ids `0..K-1`: ±0%),
Gemma-4-26B K=8 **+44%** (informed top-8 is 6% of 128 experts yet covers
half of all routed selections), OLMoE +19%.
`HOT_MODE=informed bench/bench_gptoss_hybrid.py` is the calibrate-then-pin
reference driver. Two regime laws from the same receipts
(`bench/RESULTS-gptoss-hybrid-ab.md`): the hybrid wins where the host CPU is
weak and VRAM is small — on a strong-CPU server, llama.cpp-style CPU compute
of the cold experts is ~an order faster than PCIe streaming — and on
multi-socket hosts **pin the process affinity** (`taskset` was worth 6.9× on
our cold-stream decode and 3.2× on llama.cpp's CPU-MoE in the same
measurements). The partition is math-identical to the reference forward
(both stacks decode the same NF4 values through the same kernel;
correctness-gated in the suite).

```python
from experts4bit_qlora import enable_hot_residency
# hot_sets[i] = hot expert ids for the i-th MoE layer, from a routing histogram
enable_hot_residency(model, hot_sets, device="cuda")
```

Runs on a **stock** `pip install bitsandbytes` today — see "Relationship to bitsandbytes" below.
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
[support matrix](docs/support_matrix.md).

> **Troubleshooting — OOM loading in 4-bit?** If you used
> `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))`
> and ran out of memory, that path quantized only the `nn.Linear` layers and skipped the fused
> experts ([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849))
> — they are still in bf16. Switch to `load_moe_4bit_streaming` (above), then call
> `verify_moe_4bit(model, strict=True)`: it raises and lists any expert stack still left in high
> precision, so you can confirm the fix.

## Storage modes: the support matrix

One knob selects the frozen experts' storage: `quant_type=` in code, `QUANT_TYPE=` in the
train/infer scripts — the same validation path, checked **before any checkpoint I/O**. Canonical
names are the six below; `bfloat16`/`float16` are accepted aliases (case/whitespace-insensitive);
anything else raises listing the valid set. There is no per-expert mode mixing: one module, one
scheme.

**What the words mean.** *Supported* means tested under the stated conditions — no more: an
exposed code path is not a warranty. *Experimental* means the path exists but may change or break
(the `ExpertsNbit` primitive as a whole carries upstream's experimental tag until
bitsandbytes#1965 settles). *Unsupported* means it fails loudly by design — never a silent no-op.

| Mode | Status | Intended use | Memory | Quality risk¹ | Training | Inference | Offload | Notes |
|---|---|---|---|---|---|---|---|---|
| `nf4` | supported + benchmarked | the QLoRA default | 4x smaller | ~0.17 (cap 0.25) | end-to-end (20-step test + convergence run) | fast-path + probed GEMV + prefetch | tested + benchmarked | the headline path |
| `fp4` | supported | nf4 alternative codebook | 4x | ~0.22 (0.30) | recompute-tested | dequantize + probed GEMV | same code path as nf4 | coarser than nf4 on ~Gaussian weights |
| `int8` | supported (tested contract) | higher-fidelity frozen base | 2x | ~0.017 (0.03) | LoRA step tested | dequantize (no GEMV) | identity-tested | blockwise dynamic map — **not** LLM.int8() |
| `fp8` | supported (tested contract) | int8 alternative | 2x | ~0.045 (0.08) | LoRA step tested | dequantize | same code path as int8 | bnb e4m3 **codebook**, not torch float8; coarser than int8 (test-pinned) |
| `bf16` | supported (tested contract) | reference baseline / per-layer opt-out | 1x (none) | ~0.003 (8e-3) | LoRA step tested | dequantize | identity-tested | passthrough; no absmax buffers |
| `fp16` | supported (tested contract) | as bf16 | 1x | ~0.0004 (1e-3) | LoRA step tested | dequantize | same code path as bf16 | passthrough |

¹ Forward relerr vs a float reference on synthetic ~Gaussian expert weights, measured on CPU and
A2000 kernels (bnb 0.49.2); the parenthesized cap is the calibrated test ceiling
(`tests/test_reference_parity.py`). Not an end-task quality claim.

**"Tested contract"** = build, forward parity vs a float reference (per-scheme ceilings), a
state_dict round-trip with validated metadata, a LoRA-over-frozen-base training step with the
recompute Function on the autograd tape, and offload math-identity. Offload is identity-tested
directly on `nf4`/`int8`/`bf16`; `fp4`/`fp8`/`fp16` ride the same code paths byte-for-byte. Only
`nf4` is performance-benchmarked end-to-end — the other five are correctness-tested, not measured
for speed or end-task quality.

### What ExpertsNbit is / is not

**Is:** frozen quantized *storage* for fused expert stacks (`[num_experts, out, in]`) — a
per-expert-loop forward, quantization blocks that never cross an expert boundary, and a
recompute-in-backward projection so training holds no dequantized-expert activations.

**Is not:** grouped-GEMM (per-expert loop only, intentionally), a Transformers-wide quantization
walker, double quantization, multi-GPU/FSDP, or a speed play — on a card that already fits the
model it is strictly a memory trade (see the energy caveat above).

### Experts4bit compatibility

`Experts4bit` is the 4-bit-restricted subclass (`nf4`/`fp4` only — it rejects the 8/16-bit names
*and their aliases*) and keeps its pre-0.2 API: same constructor, same `from_float`, same
state_dict tensor keys. The loader still instantiates `Experts4bit` for 4-bit runs, so existing
`isinstance(m, Experts4bit)` checks keep working.

### Known limitations & unsupported paths

- **Checkpoint metadata:** state_dicts now embed construction metadata (scheme, blocksize, dims)
  and loads validate it — loading an `fp4` checkpoint into an `nf4`-built module raises instead
  of silently decoding against the wrong codebook (the packed bytes are shape-identical).
  Pre-metadata checkpoints load unvalidated, under both `strict` modes. *New* checkpoints into
  ≤0.2.0 code: `strict=False` works (`_extra_state` lands in `unexpected_keys`); `strict=True`
  fails loudly on the unexpected key.
- **safetensors full-module saves:** the `_extra_state` entry is a dict, which safetensors
  refuses (loudly). Filter it — `{k: v for k, v in sd.items() if not k.endswith("_extra_state")}`
  — and the save loads as a legacy (unvalidated) checkpoint. Adapter-only saves never carry it.
- **Non-checkpointed offload *training* is unsupported** and fails loudly naming the invariant
  (the shipped trainer always enables gradient checkpointing).
- **`offload_model_experts` raises when it finds no `ExpertsLoRA` modules** (changed this
  version: it used to return `[]` silently). The streaming loader likewise refuses to return a
  model on which it quantized zero expert layers.
- **GEMV is 4-bit-only** and probe-gated per configuration; the 8/16-bit schemes always decode
  via the dequantize path.
- **Loader scope** is the four architecture families under [Scope](#scope); the `ExpertsNbit`
  primitive itself is model-agnostic.

### Reading the headline memory numbers

The **7.16 GB** for Qwen3-30B-A3B (and 8.47 GB for Gemma-4-26B-A4B) is **peak GPU allocation
during a QLoRA training step with `OFFLOAD_EXPERTS=1`** on the reference A2000: roughly one
layer's experts resident plus activations/adapters, while the other ~13–15 GB of packed experts
sit in pinned CPU RAM. It is a *capability* number — fits vs doesn't fit — not a throughput
claim: the same mechanism costs ~+11 % s/step at OLMoE scale and is PCIe-bound at 26–30B scale
(0.22–0.43 tok/s decode). Method and grids: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11–§12;
environment and commit pins: [`PROVENANCE.md`](PROVENANCE.md).

### How to reproduce validation

One command, no model downloads, nonzero exit on any FAIL:

```bash
python scripts/validate_expertsnbit.py
```

```
experts4bit-qlora validate | v0.2.0 | commit <sha> | torch 2.6.0+cu124 | bnb 0.49.2 | cuda yes | NVIDIA RTX A2000 12GB
[PASS] nf4   build            0.15s
[PASS] nf4   forward_parity   relerr=0.1691 (tol 0.25)
...
[PASS] -     metadata_guard   raised ValueError: checkpoint/module config mismatch...
SUMMARY pass=37 fail=0 skip=0 -> exit 0
```

It runs the tested contract per scheme (build, forward parity, state round-trip, LoRA step,
synthetic decode sanity, offload identity) plus the checkpoint-metadata guard; SKIP lines always
say why (e.g. a host whose bitsandbytes can't quantize a scheme). The full suite is
`pip install -e ".[test]" && pytest tests/ -q`; big-model numbers reproduce via the manual
[Benchmarks](#benchmarks) scripts, not this report.

### Validation grids

experts4bit-qlora does not name a winning quantization mode — it produces a measured decision
surface (fit / fidelity / speed / portability / residency budget) with per-cell provenance. An
OLMoE-1B-7B validation grid (bundle `olmoe-qlora-grid-20260705-1351`, 3 seeds) shows a storage/
offload asymmetry — resident training exposes the memory cost of wider storage while offload
collapses the 4-bit-vs-int8 gap to ~2.4–2.7 GB — and finds int8-offload a low-VRAM/high-fidelity
training candidate *for OLMoE* (best eval 3/3 seeds). Repeating also corrected a single-run
artifact: fp4 decode is **not** faster than nf4 once sampled. Expert-streaming profiling found the
offload wall **diffuse** (no hot-static pinning justified). Qwen3-30B-A3B is a separate
scale-transfer probe: nf4 resident fits a 24 GB card, int8 resident is impractical, offload is
blocked by the pod's RAM cap. Start with
[`docs/results_summary.md`](docs/results_summary.md) and
[`docs/support_matrix.md`](docs/support_matrix.md); details in
[`OLMOE_EXPERTSNBIT_GRID`](docs/OLMOE_EXPERTSNBIT_GRID.md),
[`OLMOE_REPEAT_VALIDATION_PLAN`](docs/OLMOE_REPEAT_VALIDATION_PLAN.md),
[`MODE_DECOUPLED_ADAPTERS`](docs/MODE_DECOUPLED_ADAPTERS.md),
[`EXPERT_STREAMING_PROFILE`](docs/EXPERT_STREAMING_PROFILE.md),
[`QWEN3_30B_EXPERTSNBIT_GRID`](docs/QWEN3_30B_EXPERTSNBIT_GRID.md); apparatus in
[`RUNPOD_DISTRIBUTED_VALIDATION`](docs/RUNPOD_DISTRIBUTED_VALIDATION.md) and
[`provenance_contract`](docs/provenance_contract.md). An external review pass —
[`MEASUREMENT_AUDIT`](docs/MEASUREMENT_AUDIT.md) — recomputed every number, computed the ∅/G
quality yardstick that was latent in the bundle, and downgraded the int8-offload "best eval"
claims to confounded (a precision×placement interaction the bf16 control exposes); read it
alongside the results.

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
  (**+11 % s/step** on the OLMoE A/B). A memory optimization, not a speedup: it changes *what
  fits*, not how fast. Offloading changes tensor location, not math — unit-test-verified,
  including the gradient-checkpoint recompute path. Offloaded *training* requires gradient
  checkpointing (the shipped trainer always enables it); the unsupported non-checkpointed
  combination fails loudly rather than mis-training. Details in
  [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11.

Transfer diagnostics (default off): `E4B_OFFLOAD_STATS=1` prints per-layer H2D bandwidth, prefetch
stall/slack, and a one-shot PCIe-link + ceiling report; `E4B_OFFLOAD_ARENA=1` consolidates each
layer's four expert tensors into two per-dtype copies. What they measured on the reference host —
and why offload is PCIe-bound there — is in
[`docs/OFFLOAD-TRANSFER-NOTES.md`](docs/OFFLOAD-TRANSFER-NOTES.md).

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

What inference mode adds (all `no_grad`-only; training paths are untouched):

- **Decode fast-path** — a single-token forward skips the one-hot expert-mask machinery and its
  per-expert host syncs, looping the token's `top_k` experts with 0-d device indices.
- **Fused 4-bit GEMV** — single-row base projections go through `bnb.matmul_4bit`'s GEMV kernel,
  which reads the packed NF4 weight directly instead of materializing the dequantized expert.
  Gated by a per-configuration correctness probe — and the probe passes on **stock bitsandbytes
  0.49.x**. (4-bit only; the 8/16-bit schemes decode via the dequantize path.)
- **Prefetched expert offload** (`OFFLOAD_EXPERTS=1`, default `PREFETCH=1`) — decode with experts
  that exceed VRAM: layer `L+1`'s NF4 experts copy on a side CUDA stream while layer `L` computes.
  Staging is layer-granular, so the schedule is deterministic — no expert-prediction needed — and
  residency is bounded at two layers.

Measured on the RTX A2000 (OLMoE + the r16 adapter, 128 greedy tokens; big models: base model,
96 tokens; full grids + analysis in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §12):

| model | config | tok/s | peak GPU |
|---|---|:---:|:---:|
| OLMoE-1B-7B | resident (experts on GPU) | 3.08 | 4.86 GB |
| OLMoE-1B-7B | offload, serial | 0.40 | 1.45 GB |
| OLMoE-1B-7B | **offload + prefetch** | **1.44** | **1.68 GB** |
| Gemma-4-26B-A4B | resident | OOM | — |
| Gemma-4-26B-A4B | **offload + prefetch** | **0.43** | **6.16 GB** |
| Qwen3-30B-A3B | resident | OOM | — |
| Qwen3-30B-A3B | **offload + prefetch** | **0.22** | **4.41 GB** |

Same honest framing as training — capability, not throughput — and **the levers are
shape-dependent, measured**: at OLMoE scale prefetch is the result (3.65× over serial) and the
GEMV route is neutral; at 26–30B scale decode is so transfer-bound that prefetch's ratio shrinks
(1.36× / 1.08×), while GEMV swings from **+46 % on Gemma-4** (big per-expert stacks — avoided
dequantize traffic dominates) to **−8 % on Qwen3-30B** (thin experts — it doesn't; prefetch +
dequantize is Qwen3's best config at 0.238 tok/s). §12c scores the prediction this falsified.
Measure your model with the kill-switches; don't extrapolate across shapes.

Library users: `enable_inference_prefetch(handles)` links the offload handles the loader (or
`offload_model_experts`) returns; `load_moe_4bit_streaming(..., offload=True, prefetch=True)` does
it for you. Serve with the training run's `QUANT_TYPE`. Kill-switches for A/B:
`E4B_DECODE_FASTPATH=0`, `E4B_INFER_GEMV=0`.

## Serving over HTTP (Docker)

`experts4bit_qlora.serve` wraps the inference path in a FastAPI app so the fine-tune can be
shared by other services instead of each caller paying its own model load — built for a small
GPU that has other tenants. With `OFFLOAD_EXPERTS=1` (the serve default) the OLMoE endpoint
idles at **~1.7 GB GPU**; requests are batch-1 and queue behind a single GPU worker (the offload
residency machinery is deliberately single-flight), so this is an *availability* deployment, not
a throughput one.

```bash
pip install "experts4bit-qlora[serve]"
E4B_ADAPTERS="alpaca=./out/adapter_best.pt" python -m experts4bit_qlora.serve   # 0.0.0.0:8777
```

**Many fine-tunes, one base.** Every adapter in `E4B_ADAPTERS` (plus `base`, the un-tuned model)
is served concurrently over the same NF4 base: adapters live in pinned CPU RAM and hot-swap over
the live LoRA parameters per request (~tens of ms against a multi-second generation), validated
at startup against the model's LoRA key-set — so N fine-tunes cost the VRAM of one. All adapters
must share the server's `R`/`ALPHA` (R is checked structurally; ALPHA is invisible in a `.pt`).

- `POST /generate` — `{prompt, adapter?, max_new_tokens?, temperature?, top_p?,
  repetition_penalty?, stream?, seed?}` → `{text, adapter, tokens, tok_per_s, swap_ms, stopped}`,
  or SSE token events with `stream: true`.
- `GET /health` — status, adapters, queue depth, GPU memory; never blocks behind a generation.
- `POST /v1/completions` + `GET /v1/models` — OpenAI-compatible (`model` selects the adapter).
  Deliberately no `/v1/chat/completions`: OLMoE has no chat template; send Alpaca-format prompts
  (`### Instruction:\n...\n\n### Response:\n`).

Guardrails: `E4B_QUEUE_MAX` waiting requests (then 503 + Retry-After), `E4B_MAX_INPUT_TOKENS`
(413), `E4B_MAX_NEW_TOKENS` clamp, `E4B_REQUEST_TIMEOUT_S` (partial text, `stopped: "timeout"`).
The allocator cache is released to the driver between requests (`E4B_EMPTY_CACHE=1`) so bursty
GPU neighbors can use the headroom.

[`deploy/`](deploy/) has the Dockerfile + compose file (CUDA 12.4 runtime base, the pinned stack
the A2000 numbers were measured with). One deployment note that costs 3.6× if missed: the
container needs `ulimits: memlock: -1` — without it the pinned-RAM homes silently fall back to
pageable and offloaded decode drops from 1.44 to ~0.4 tok/s.

## Benchmarks

```bash
# Runs on stock bitsandbytes:
python bench/bench_energy_excluded.py                    # memory wall + tokens-per-joule vs batch

# Require bitsandbytes >= 0.50 — measure the upstream matmul_4bit routing (#1965):
python bench/_upstream/bench_matmul4bit.py --mode both   # equivalence + latency/memory
python bench/_upstream/bench_energy.py                   # joules/op: bf16 vs dequant vs matmul_4bit
```

The LoRA-placement ablation (which of experts / attention / router to train) and full energy
analysis are written up in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Short version: on Alpaca
the placements are largely **redundant**, attention-only is the efficiency pick, and training the
router **hurts**.

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

`ExpertsNbit` / `Experts4bit` are bitsandbytes primitives, proposed upstream in
[bitsandbytes#1965](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965). Until that
ships in a release, this package **vendors** a copy (`experts4bit_qlora/_vendor/experts.py`) so it
runs on stock bitsandbytes today. The import shim prefers the upstream classes when present *and
still satisfying everything this package promises about them*: the internals `ExpertsLoRA` builds
on, `Experts4bit` a subclass of `ExpertsNbit`, and the state_dict metadata contract
(`get`/`set_extra_state` overrides). Both names must resolve to the same implementation, never a
mix — and anything less falls back to the vendored copy:

```python
try:
    from bitsandbytes.nn import Experts4bit, ExpertsNbit   # once bitsandbytes#1965 releases (if compatible)
except ImportError:
    from ._vendor.experts import Experts4bit, ExpertsNbit  # vendored fallback (stock bnb)
```

Nothing in training depends on the bitsandbytes version: the recompute-in-backward projection
delivers the activation-memory win on any release. The only `bnb.matmul_4bit` use left in the
package is the inference decode GEMV, which is probe-gated per configuration and passes on stock
0.49.x. When #1965 lands upstream: bump the `bitsandbytes` floor and delete `_vendor/` — no API
change.

### Prior art

The closest public prior art is
[woct0rdho/transformers-qwen3-moe-fused](https://github.com/woct0rdho/transformers-qwen3-moe-fused)
(Apache-2.0, June 2025 — a year before this package existed), which demonstrated bnb 4-bit
quantization of the fused 3-D expert stack and per-expert *stacked* LoRA
(`[num_experts, r, in]` / `[num_experts, out, r]`) for Qwen3-MoE, wrapped around a Triton
grouped-GEMM forward, and has since fed the Transformers-5-era fused-MoE ecosystem (Transformers,
PEFT, Unsloth). It reached the core primitive — 4-bit on the fused stack with trainable per-expert
adapters — first, and it is the better choice when fused-forward throughput is the goal. The two
projects optimize different axes: that one is a *kernel* project (grouped-GEMM speed; its fused
4-bit-dequant kernel is forward-only and listed as in-progress); this one is a *storage-contract*
project — deliberately per-expert-loop (see
[what ExpertsNbit is / is not](#what-expertsnbit-is--is-not)) — whose distinct contributions are
the tested training contract (recompute-in-backward holding no dequantized-expert activations,
offload asserted bit-identical to resident execution, packed storage asserted unchanged through
training steps), the fidelity-pinned N-bit storage matrix, the streaming loader + past-VRAM expert
offload, and train/serve byte-identity (the served base is asserted `torch.equal` to the base the
adapters were trained against).

## Provenance & audits

Every measured number above traces to a committed script/test, an exact environment, and a repo
commit in [`PROVENANCE.md`](PROVENANCE.md) — and that file is OpenTimestamps-anchored: `ots verify
PROVENANCE.md.ots PROVENANCE.md` checks the on-disk bytes against the calendar proof, the footer
carries the hash-chain of prior revisions, and superseded proofs are retained in
[`.ots-history/`](.ots-history/). Falsification work lives under [`audits/`](audits/) — most
recently the audit of unsloth-zoo's MoE-4bit fix that produced unsloth-zoo#849/#850
([`audits/unsloth-zoo-4032/REPORT.md`](audits/unsloth-zoo-4032/REPORT.md)).

## License

MIT (see [LICENSE](LICENSE)). `experts4bit_qlora/_vendor/experts.py` is vendored from
bitsandbytes (also MIT) pending upstream merge.
