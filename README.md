# experts4bit-qlora

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
- **It scales past VRAM (`OFFLOAD_EXPERTS=1`).** The frozen 4-bit experts — the bulk of the
  weights — can be streamed from **pinned CPU RAM** one layer at a time, GPU-resident only for that
  layer's forward (and its gradient-checkpoint recompute) and evicted after. Peak GPU drops by
  roughly *(experts footprint − one layer)*, so a fused-MoE whose 4-bit experts exceed the card
  (Qwen3-30B-A3B ~15 GB, Gemma-4-26B-A4B ~13 GB) can QLoRA-train on 12 GB, at the cost of one PCIe
  transfer per layer per pass. Same memory-for-compute trade as above: it changes *what fits*, not
  speed. Offloading changes tensor location, not math — unit-test-verified, including the
  gradient-checkpoint recompute path (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11).
- **Honest caveat — this is a memory technology, not an energy one.** On a GPU that *already*
  fits the model, 4-bit is a **1.2–2.3× energy penalty** (NF4 is storage-only; the GEMM runs in
  bf16 either way, plus dequant). The energy win only shows up when memory is the binding
  constraint — then it's the difference between running and not, and up to **4.4× lower
  energy/token** from the batch that freed memory unlocks. Numbers and method in the docs.

## Install

```bash
pip install -e .            # primitive + adapters + benchmarks (torch + bitsandbytes only)
pip install -e ".[train]"   # + the OLMoE streaming trainer (transformers>=5.0, datasets, ...)
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

The `Experts4bit` primitive and `ExpertsLoRA` adapters are **model-agnostic** — they operate on any
fused `[num_experts, out, in]` expert stack. The **streaming loader / trainer**
(`python -m experts4bit_qlora.train`) is currently **OLMoE-specific**: it assumes OLMoE's checkpoint
key layout, `OlmoeRotaryEmbedding`, and `OlmoeAttention`, and **fails fast with a clear error** on
other architectures. Other fused-MoE models (e.g. Qwen3-MoE) need a loader adaptation — PRs welcome.

**Expert CPU-offload** (`OFFLOAD_EXPERTS=1`) is orthogonal to the loader: the streaming/eviction
mechanism (`experts4bit_qlora/offload.py`) is model-agnostic — it hooks any `ExpertsLoRA` — so it
works for whatever architectures the loader supports. Its correctness is validated here by unit
tests (offload = location, not math, including the gradient-checkpoint recompute path); the
peak-memory-drop / throughput A/B ([`bench/run-offload-ab.sh`](bench/run-offload-ab.sh), OLMoE) runs
on the card, and the measured 26–35B-on-12 GB headline lands once a loader for those architectures
does. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §11.

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
