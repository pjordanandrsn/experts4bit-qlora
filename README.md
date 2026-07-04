# experts4bit-qlora

[![CI](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/experts4bit-qlora/actions/workflows/ci.yml)

QLoRA fine-tuning of **fused Mixture-of-Experts** weights on a single small GPU — the part that
doesn't fit anywhere else yet.

## The problem

transformers v5 stores MoE experts as one fused 3-D `nn.Parameter` per layer
(`OlmoeExperts`, `Qwen3MoeExperts`, …). bitsandbytes' 4-bit walker only replaces `nn.Linear`
modules, so it **silently skips the experts** — which are the overwhelming majority of a MoE's
weights. `load_in_4bit` "shrinks" the model but the experts stay in full precision
([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)).

`Experts4bit` is the primitive that 4-bit-quantizes exactly that fused stack. This package pairs
it with a **streaming loader** and **per-expert LoRA**, so you can actually *fine-tune* a real
sparse-MoE on reasonable hardware.

## What it buys you (measured on an RTX A2000 12 GB)

- **It fits at all.** Full bf16 OLMoE-1B-7B is ~13.9 GB — it **OOMs** on a 12 GB card. In 4-bit
  it loads at **4.70 GB** and trains in <8 GB. The streaming loader never materializes the bf16
  model in CPU *or* GPU RAM (verified under a 3 GB container RAM cap).
- **It trains.** QLoRA on the frozen NF4 experts improves a held-out Alpaca eval from
  **1.4813 → 1.0290** (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)).
- **It scales past VRAM (`OFFLOAD_EXPERTS=1`).** The frozen 4-bit experts — the bulk of the
  weights — can be streamed from **pinned CPU RAM** one layer at a time, GPU-resident only for that
  layer's forward (and its gradient-checkpoint recompute) and evicted after. Peak GPU drops by
  roughly *(experts footprint − one layer)*, so a fused-MoE whose 4-bit experts exceed the card
  (Qwen3-30B-A3B ~15 GB, Gemma-4-26B-A4B ~13 GB) can QLoRA-train on 12 GB — **both measured on an
  RTX A2000** (peak **7.16 GB** / **8.47 GB**; both OOM *without* offload) — at the cost of one PCIe
  transfer per layer per pass. Same memory-for-compute trade as above: it changes *what fits*, not
  speed. Offloading changes tensor location, not math — unit-test-verified, including the
  gradient-checkpoint recompute path (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11).
- **It serves the fine-tune it made (`python -m experts4bit_qlora.infer`).** The adapters run
  over the *exact* NF4 base they were trained against — no GGUF/AWQ re-quantization shifting the
  error surface. With `OFFLOAD_EXPERTS=1` + prefetch, OLMoE **decodes at 1.44 tok/s in 1.68 GB**
  on the A2000 (3.65× over serial offload; resident is 3.08 tok/s at 4.86 GB) — see
  [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §12.
- **Honest caveat — this is a memory technology, not an energy one.** On a GPU that *already*
  fits the model, 4-bit is a **1.2–2.3× energy penalty** (NF4 is storage-only; the GEMM runs in
  bf16 either way, plus dequant). The energy win only shows up when memory is the binding
  constraint — then it's the difference between running and not, and up to **4.4× lower
  energy/token** from the batch that freed memory unlocks. Numbers and method in the docs.

## Install

```bash
pip install experts4bit-qlora           # primitive + adapters + benchmarks (torch + bitsandbytes)
pip install "experts4bit-qlora[train]"  # + the streaming MoE trainer (transformers>=5.0, datasets, ...)
```

Runs on a **stock** `pip install bitsandbytes` today — see "Relationship to bitsandbytes" below.

## Quickstart

```python
import torch
from experts4bit_qlora import Experts4bit, ExpertsLoRA

# Freeze a fused expert stack in 4-bit, attach trainable per-expert LoRA.
gate_up = torch.randn(8, 2 * 256, 128)          # [num_experts, 2*intermediate, hidden]
down    = torch.randn(8, 128, 256)              # [num_experts, hidden, intermediate]
base    = Experts4bit.from_float(gate_up, down, quant_type="nf4", compute_dtype=torch.float32)
model   = ExpertsLoRA(base, r=8, alpha=16)      # only the LoRA adapters train
```

End-to-end OLMoE QLoRA fine-tune (needs a CUDA GPU + `[train]` extras):

```bash
STEPS=150 R=8 TRAIN_EXPERTS=1 TRAIN_ATTENTION=0 OUT=./out \
  python -m experts4bit_qlora.train
```

Add `OFFLOAD_EXPERTS=1` to keep the frozen 4-bit experts in pinned CPU RAM and stream one layer to
the GPU at a time (set `OFFLOAD_PIN=0` to skip pinning). Lowers peak GPU memory at a per-layer PCIe
cost — the way to fit experts that exceed VRAM.

## Scope

The `Experts4bit` primitive and `ExpertsLoRA` adapters are **model-agnostic**. The **streaming loader /
trainer** (`python -m experts4bit_qlora.train`) supports SwiGLU fused-MoE architectures — experts stored
either **per-expert** or already-**fused** on disk:

- **OLMoE** (OLMoE-1B-7B) — convergence-tested end-to-end; fits a 12 GB card at ~4.7 GB.
- **Qwen3-MoE / Qwen3.5-MoE** — same checkpoint + module layout as OLMoE (verified byte-identical);
  structurally tested in `tests/test_loader_architectures.py`. The real weights (30–35B) need a
  ≥24 GB card — or the **CPU-offloading path** (`OFFLOAD_EXPERTS=1`, below) — to fit 12 GB.
- **Gemma-4 (text tower)** — different internally (experts at `layers.{i}.experts` beside a parallel
  dense MLP + a custom router; experts fused on disk) — handled and structurally tested.

All three are covered by `tests/test_loader_architectures.py`. Real Qwen3/Gemma weights (26–35B) need a
≥24 GB card — or the CPU-offloading path (`OFFLOAD_EXPERTS=1`, below) — to fit 12 GB. Unsupported
architectures **fail fast with a clear error**; PRs for more welcome.

**Expert CPU-offload** (`OFFLOAD_EXPERTS=1`) is orthogonal to the loader: the streaming/eviction
mechanism (`experts4bit_qlora/offload.py`) is model-agnostic — it hooks any `ExpertsLoRA` — so it
works for whatever architectures the loader supports. Its correctness is validated here by unit
tests (offload = location, not math, including the gradient-checkpoint recompute path); the
peak-memory-drop / throughput A/B ([`bench/run-offload-ab.sh`](bench/run-offload-ab.sh), OLMoE) runs
on the card. Since the loader supports **Qwen3-MoE** and **Gemma-4**, offload also fits
**Qwen3-30B-A3B** and **Gemma-4-26B-A4B** on 12 GB — measured on the A2000 (peak **7.16** / **8.47 GB**;
both OOM without offload). See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11.

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
  which reads the packed NF4 weight directly instead of materializing the dequantized expert
  (~4x less memory traffic per expert at decode). Gated by a per-configuration correctness probe —
  and the probe passes on **stock bitsandbytes 0.49.x**, where the multi-row training route is
  correctly refused.
- **Prefetched expert offload** (`OFFLOAD_EXPERTS=1`, default `PREFETCH=1`) — decode with experts
  that exceed VRAM: layer `L+1`'s NF4 experts copy on a side CUDA stream while layer `L` computes.
  Staging is layer-granular, so the schedule is deterministic — no expert-prediction needed, unlike
  expert-granular prefetch systems — and residency is bounded at two layers. The last layer
  prefetches the first: exactly the next token's first need.

Measured on the RTX A2000 (OLMoE + the r16 adapter, 128 greedy tokens; full grid + analysis in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §12):

| config | tok/s | peak GPU |
|---|:---:|:---:|
| resident (experts on GPU) | 3.08 | 4.86 GB |
| offload, serial | 0.40 | 1.45 GB |
| **offload + prefetch** | **1.44** | **1.68 GB** |

Same honest framing as training: this is *capability* (generate with a fused-MoE + live adapters in
1.68 GB, where bf16 OOMs), not throughput. Decode under offload is PCIe-bound; **prefetch is the
lever that matters** (3.65× over serial — the side-stream copy overlaps the entire next layer's
compute, and the layer-granular schedule needs no expert prediction). The GEMV route and fast-path
measure *neutral* at OLMoE scale — they are correctness/portability features, not speedups.

Library users: `enable_inference_prefetch(handles)` links the offload handles the loader (or
`offload_model_experts`) returns; `load_moe_4bit_streaming(..., offload=True, prefetch=True)` does
it for you. Kill-switches for A/B: `E4B_DECODE_FASTPATH=0`, `E4B_INFER_GEMV=0`.

## Benchmarks

```bash
# Runs on stock bitsandbytes (uses the portable dequantize forward):
python bench/bench_energy_excluded.py                    # memory wall + tokens-per-joule vs batch

# Require bitsandbytes >= 0.50 — measure the upstream matmul_4bit optimization (#1965):
python bench/_upstream/bench_matmul4bit.py --mode both   # equivalence + latency/memory
python bench/_upstream/bench_energy.py                   # joules/op: bf16 vs dequant vs matmul_4bit
```

The LoRA-placement ablation (which of experts / attention / router to train) and full energy
analysis are written up in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Short version: on Alpaca
the placements are largely **redundant**, attention-only is the efficiency pick, and training the
router **hurts**.

## Relationship to bitsandbytes

`Experts4bit` is a bitsandbytes primitive, proposed upstream in
[bitsandbytes#1965](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965). Until it
ships in a release, this package **vendors** a copy (`experts4bit_qlora/_vendor/experts.py`) so it
runs on stock bitsandbytes today. The import shim prefers the upstream class when present *and
still exposing the internals `ExpertsLoRA` builds on* (so a reviewed/diverged upstream merge can't
silently break the adapters — the shim falls back to the vendored copy instead):

```python
try:
    from bitsandbytes.nn import Experts4bit     # once bitsandbytes#1965 releases (if compatible)
except ImportError:
    from ._vendor.experts import Experts4bit    # vendored fallback (stock bnb)
```

The vendored forward also **auto-detects** whether `matmul_4bit` is correct on your installed
bitsandbytes — it only handles this weight layout correctly on **bnb ≥ 0.50**, so on older releases
the primitive uses the portable dequantize path, and the `matmul_4bit` memory optimization engages
automatically once you upgrade. Results are correct on any supported bnb either way.

When it lands upstream: bump the `bitsandbytes` floor and delete `_vendor/` — no API change.

## License

MIT (see [LICENSE](LICENSE)). `experts4bit_qlora/_vendor/experts.py` is vendored from
bitsandbytes (also MIT) pending upstream merge.
