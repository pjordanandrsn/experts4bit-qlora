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
  it loads at **4.69 GB** and trains in <8 GB. The streaming loader never materializes the bf16
  model in CPU *or* GPU RAM (verified under a 3 GB container RAM cap).
- **It trains.** QLoRA on the frozen NF4 experts improves a held-out Alpaca eval from
  **1.4813 → 1.0290** (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)).
- **Honest caveat — this is a memory technology, not an energy one.** On a GPU that *already*
  fits the model, 4-bit is a **1.2–2.3× energy penalty** (NF4 is storage-only; the GEMM runs in
  bf16 either way, plus dequant). The energy win only shows up when memory is the binding
  constraint — then it's the difference between running and not, and up to **4.4× lower
  energy/token** from the batch that freed memory unlocks. Numbers and method in the docs.

## Install

```bash
# primitive + adapters + benchmarks (torch + bitsandbytes):
pip install "git+https://github.com/pjordanandrsn/experts4bit-qlora"
# + the streaming MoE trainer (transformers>=5.0, datasets, ...):
pip install "experts4bit-qlora[train] @ git+https://github.com/pjordanandrsn/experts4bit-qlora"
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

## Scope

The `Experts4bit` primitive and `ExpertsLoRA` adapters are **model-agnostic**. The **streaming loader /
trainer** (`python -m experts4bit_qlora.train`) supports fused-MoE architectures that store experts
per-expert on disk under `model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight` with a SwiGLU gate:

- **OLMoE** (OLMoE-1B-7B) — convergence-tested end-to-end; fits a 12 GB card at ~4.7 GB.
- **Qwen3-MoE / Qwen3.5-MoE** — same checkpoint + module layout (verified byte-for-byte identical to
  OLMoE's on-disk format); structurally tested in `tests/test_loader_architectures.py`. The real weights
  (30–35B) need a ≥24 GB card — or the CPU-offloading path (tracked separately) — to fit 12 GB.

Anything else **fails fast with a clear error**. **Gemma 4** is a genuinely different design (experts at
`layers.{i}.experts` beside a parallel dense MLP, with a custom router) and needs its own loader
adaptation — not yet supported. PRs welcome.

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
runs on stock bitsandbytes today. The import shim prefers the upstream class when present:

```python
try:
    from bitsandbytes.nn import Experts4bit      # once bitsandbytes#1965 releases
except ImportError:
    from ._vendor.experts import Experts4bit     # vendored fallback (stock bnb)
```

The vendored forward also **auto-detects** whether `matmul_4bit` is correct on your installed
bitsandbytes — it only handles this weight layout correctly on **bnb ≥ 0.50**, so on older releases
the primitive uses the portable dequantize path, and the `matmul_4bit` memory optimization engages
automatically once you upgrade. Results are correct on any supported bnb either way.

When it lands upstream: bump the `bitsandbytes` floor and delete `_vendor/` — no API change.

## License

MIT (see [LICENSE](LICENSE)). `experts4bit_qlora/_vendor/experts.py` is vendored from
bitsandbytes (also MIT) pending upstream merge.
