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
```

Runs on a **stock** `pip install bitsandbytes` today — see "Relationship to bitsandbytes" below.

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
- **Loader scope** is the three architectures under [Scope](#scope); the `ExpertsNbit` primitive
  itself is model-agnostic.

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

All three are covered by `tests/test_loader_architectures.py`. Real Qwen3/Gemma weights (26–35B)
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
