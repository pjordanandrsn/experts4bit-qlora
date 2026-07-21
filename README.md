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
  **1.4813 → 1.0290** (see [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/METHODOLOGY.md)).
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
  (receipts: [`bench/RESULTS-informed-hotsets.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/bench/RESULTS-informed-hotsets.md)).
  K=0 streams everything; K=all is fully resident; the middle is yours to trade.
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


### Which door? (all six, one line each)

| You want | Call | Status |
|---|---|---|
| Train / maximum compatibility (any host, any scheme) | nothing — the reference `ExpertsNbit` forward is the default | supported + convergence-tested |
| Faster frozen-expert **inference** on CUDA | `enable_fast(model)` (`[fast]`) | supported + benchmarked (3.65× bs=1) |
| Serve past VRAM: hot experts resident, cold **streamed** | `enable_pipelined_residency(model, hot_sets, k_slots=k)` (`[fast]`) | supported — the current serving engine (K is config: empty set = pure streaming, all experts = fully resident) |
| Same, the v0 engine | `enable_hot_residency(model, hot_sets)` (`[fast]`) | **superseded** by pipelined (kept through 0.6 to reproduce the v0 receipts; removal in 0.7; warns at call) |
| Hot experts resident, cold **computed on the host CPU** | `enable_cold_engine(model, hot_sets, dequant="auto")` | correct + CPU-complete tests (bit-exact host decode; all-cold runs with no CUDA/`[fast]`); **performance-experimental** — the host decode is a correctness path until the AVX2 kernel lands |
| Models whose experts exceed VRAM, training or serving | `OFFLOAD_EXPERTS=1` / `load_moe_4bit_streaming(..., offload=True, prefetch=True)` | supported + benchmarked (layer-granular, deterministic) |

```python
from experts4bit_qlora import enable_fast, disable_fast
enable_fast(model)    # returns the number of expert modules patched
```

**Hot-expert residency** (`enable_hot_residency` — **superseded by
`enable_pipelined_residency`**, kept through 0.6 to reproduce the v0 receipts;
removal in 0.7; needs `[fast]` — it runs on
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

**Cold engine** (`enable_cold_engine`) is the other side of that regime law:
the same hot partition stays resident on the GPU, but the cold tail is
**computed on the host** from the CPU-resident NF4 — per-token traffic is
activation-sized, never weight-sized (the `--n-cpu-moe` regime at expert
rather than layer granularity, for the strong-CPU hosts where the hybrid A/B
receipts put CPU compute ~an order over PCIe streaming). The host decode is
bit-exact against bitsandbytes' CPU `dequantize_4bit` and backend-selected
around its AVX2 cliff: `dequant="auto"` takes bnb's AVX-512 kernel only where
`avx512f` is present and otherwise a pure-torch decode (on AVX2-only hosts
bnb silently falls back below even naive torch — grouped-nf4-gemm
`bench/cold-engine/` receipts). An all-cold configuration (`hot_sets` of
empties, `device="cpu"`) is a pure-host MoE and needs neither CUDA nor
`[fast]`.

```python
from experts4bit_qlora import enable_cold_engine
enable_cold_engine(model, hot_sets, device="cuda", dequant="auto")
```

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
[support matrix](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/support_matrix.md).

> **Troubleshooting — OOM loading in 4-bit?** If you used
> `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))`
> and ran out of memory, that path quantized only the `nn.Linear` layers and skipped the fused
> experts ([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849))
> — they are still in bf16. Switch to `load_moe_4bit_streaming` (above), then call
> `verify_moe_4bit(model, strict=True)`: it raises and lists any expert stack still left in high
> precision, so you can confirm the fix.

## Storage modes: the support matrix

Moved to **[docs/STORAGE-MODES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/STORAGE-MODES.md)** — The full storage-mode support matrix (ExpertsNbit vs Experts4bit, compatibility, known limitations, the headline-number reading, reproduction + validation grids).

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
  [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/METHODOLOGY.md) §11.

Transfer diagnostics (default off): `E4B_OFFLOAD_STATS=1` prints per-layer H2D bandwidth, prefetch
stall/slack, and a one-shot PCIe-link + ceiling report; `E4B_OFFLOAD_ARENA=1` consolidates each
layer's four expert tensors into two per-dtype copies. What they measured on the reference host —
and why offload is PCIe-bound there — is in
[`docs/OFFLOAD-TRANSFER-NOTES.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/OFFLOAD-TRANSFER-NOTES.md).

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
96 tokens; full grids + analysis in [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/METHODOLOGY.md) §12).
*These are v0 offload-path figures; the pipelined engine supersedes them for decode — see the
dial and [`bench/RESULTS-informed-hotsets.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/bench/RESULTS-informed-hotsets.md).*

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

Moved to **[docs/SERVING.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/SERVING.md)** — The FastAPI serving shim + Docker deployment (endpoints, env knobs, the localhost-by-default / E4B_TOKEN posture).

## Benchmarks

Moved to **[docs/BENCHMARKS.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/BENCHMARKS.md)** — The benchmark scripts and how to run them (memory wall, tokens-per-joule, the upstream matmul_4bit comparison).

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

Moved to **[docs/BITSANDBYTES.md](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/BITSANDBYTES.md)** — How ExpertsNbit/Experts4bit relate to bitsandbytes #1965, the vendored-copy shim, and the prior-art credits.

## Provenance & audits

Every measured number above traces to a committed script/test, an exact environment, and a repo
commit in [`PROVENANCE.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/PROVENANCE.md) — and that file is OpenTimestamps-anchored: `ots verify
PROVENANCE.md.ots PROVENANCE.md` checks the on-disk bytes against the calendar proof, the footer
carries the hash-chain of prior revisions, and superseded proofs are retained in
[`.ots-history/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.6.3/.ots-history/). Falsification work lives under [`audits/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.6.3/audits/) — most
recently the audit of unsloth-zoo's MoE-4bit fix that produced unsloth-zoo#849/#850
([`audits/unsloth-zoo-4032/REPORT.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/audits/unsloth-zoo-4032/REPORT.md)).

## License

MIT (see [LICENSE](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/LICENSE)). `experts4bit_qlora/_vendor/experts.py` is vendored from
bitsandbytes (also MIT) pending upstream merge.
