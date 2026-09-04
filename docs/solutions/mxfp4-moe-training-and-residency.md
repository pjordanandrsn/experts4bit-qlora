# How do I train and serve MoE models released in MXFP4 (gpt-oss, DeepSeek-V4)?
<!-- summary: Choose between the convenient QLoRA path, which decodes MXFP4 and re-quantises to NF4, and the native-byte path, which keeps the released blocks and scales in an arena. -->

`load_moe_4bit_streaming` dequantises the released MXFP4 experts bit-identically and, by default, re-quantises them to NF4 for QLoRA. To keep computing on the released bytes, relocate them into an arena with `grouped-nf4-gemm` and bind it with `enable_mxfp4_nvme_residency` (serving) or `enable_nvme_train_residency` (training); the native MXFP4 expert store for the paged engine is a separate opt-in whose quality gate is still open.

## Two fidelity paths — choose before loading

Only the second path keeps the checkpoint's original expert bytes.

1. **Convenient QLoRA path — decode, then re-quantise to NF4.** The quantising branch of `load_moe_4bit_streaming` (`experts4bit_qlora/loader.py`) reads the released blocks and scales through `experts4bit_qlora.formats.mxfp4.dequantize_mxfp4` — verified bit-identical to transformers' reference decode in `tests/test_mxfp4_dequant.py` — and then builds the expert stack in the storage you asked for, NF4 by default: `GptOssExperts4bit.from_gptoss(..., quant_type=quant_type)` for gpt-oss and `DeepseekV4Experts4bit.from_deepseek_v4(..., quant_type=quant_type)` for DeepSeek-V4, each of which quantises the decoded stack through `Experts4bit.from_float`. What trains and serves afterwards is an NF4 re-quantisation of an exact decode of the release: the decode is bit-exact, the NF4 that follows it is the quantiser's output, and that — not the release — is the provenance of the served experts. Needs no arena and no kernel package. `verify_moe_4bit(model, strict=True)` proves the stack is 4-bit, not that it is the released bytes.
2. **Native-byte path — retain the released MXFP4 blocks and scales.** Relocate the checkpoint's own blocks and scales verbatim into an arena with `nvme_arena.bake_expert_tensors` (hash-preserving; the manifest's `bake_mode` records it) and bind it with `enable_mxfp4_nvme_residency` (serving through `mxfp4_grouped`'s native kernels) or `enable_nvme_train_residency` on an `arena=..., arena_train=True` load (training against the arena, gradient checkpointing required). The paged engine's native MXFP4 store — `enable_serve_experts_int4` on gpt-oss, which never re-quantises onto the int4 grid — is the all-VRAM form of the same idea. Here the checkpoint's expert bytes are what computes, and provenance is preserved end to end.

## Symptoms

- "gpt-oss MXFP4 experts: training and residency" / "QLoRA on an MXFP4 checkpoint" / "MXFP4 expert inference kernel".
- The checkpoint ships `gate_up_proj_blocks` / `gate_up_proj_scales` (gpt-oss) or `I8` blocks with `F8_E8M0` scales (DeepSeek-V4), and a naive `from_pretrained` materialises them to bf16 first.
- `KeyError: 'F8_E8M0'` when staging a DeepSeek-V4 arena for training.

## Why it happens

MXFP4 (OCP microscaling FP4) stores two e2m1 nibbles per byte in blocks of 32 values sharing one e8m0 power-of-two scale. gpt-oss adds per-expert biases, interleaved gate/up rows and a clamped GLU; DeepSeek-V4 keeps SwiGLU with one-sided and two-sided clamps and stores its *dense* half as block-scaled FP8 ([`../DEEPSEEK-V4.md`](../DEEPSEEK-V4.md)). Re-quantising these experts to a uniform int4 grid fails the quality gate because NF4's levels sit on e2m1's and int4's do not ([`../SERVING-THROUGHPUT.md`](../SERVING-THROUGHPUT.md)).

## Which project solves it

**experts4bit-qlora** owns the model side: `formats.mxfp4.dequantize_mxfp4` (verified bit-identical to transformers' reference, `tests/test_mxfp4_dequant.py`), `arch/gptoss.py` (biases, de-interleave, clamped GLU, `GptOssExperts4bit`), `arch/deepseek_v4.py` (key map, FP8 dense side, `_apply_gate` epilogue), and the binding engines `engines/nvme_experts.py`, `engines/nvme_train.py` and `engines/int4_experts.py`. **grouped-nf4-gemm** ([GitHub](https://github.com/pjordanandrsn/grouped-nf4-gemm), [PyPI](https://pypi.org/project/grouped-nf4-gemm/)) owns the native MXFP4 kernels (`mxfp4_grouped`, including the decode GEMV), the relocation bake `nvme_arena.bake_expert_tensors`, and `mxfp4_residency`. Kernel-level questions belong there; `[fast]` is the seam.

## Install

```bash
pip install "experts4bit-qlora[train]"   # minimum/reference training: the loader (decode-then-NF4 path)
pip install "experts4bit-qlora[fast]"    # residency/NVMe/fast-kernel route: grouped-nf4-gemm's MXFP4 kernels, arena bake, residency
```

## Smallest correct example

Needs: GPU + network + model download + local NVMe.

```bash
python -c "
from nvme_arena import bake_expert_tensors
from mxfp4_residency import V4_RESIDENCY_KINDS
bake_expert_tensors('/path/to/DeepSeek-V4-Flash', '/nvme/v4.mxarena',
                    name_template='layers.{layer}.ffn.experts.{expert}.{kind}',
                    kinds=V4_RESIDENCY_KINDS)"        # relocation: no re-quantisation
```

Needs: the same GPU, the `[train]` and `[fast]` extras, and the arena baked above.

```python
import torch
from experts4bit_qlora import enable_mxfp4_nvme_residency, load_moe_4bit_streaming

model, cfg = load_moe_4bit_streaming("deepseek-ai/DeepSeek-V4-Flash", "cuda", torch.bfloat16,
                                     r=8, alpha=16, quant_type="nf4", arena="/nvme/v4.mxarena")
n = enable_mxfp4_nvme_residency(model, "/nvme/v4.mxarena",
                                k_slots=cfg.num_experts_per_tok, hot_rows=16)
assert n > 0
```

Training against the same relocated bytes: load with `arena=..., arena_train=True`, then `enable_nvme_train_residency(model, arena, hot_rows=<expert count>)` with gradient checkpointing enabled ([`offload-moe-experts-to-cpu-or-nvme.md`](offload-moe-experts-to-cpu-or-nvme.md)). The default NF4 path needs no arena: `load_moe_4bit_streaming("openai/gpt-oss-20b", ...)` then `verify_moe_4bit(model, strict=True)`.

## Expected result

`enable_mxfp4_nvme_residency` returns the number of MoE modules bound (one per layer); a model loaded with `arena=` has its experts on `meta` and cannot run until an engine is bound. `enable_nvme_train_residency` returns the number of modules moved and refuses an undersized `hot_rows`. For the NF4 path, `verify_moe_4bit(model, strict=True)` returns without raising. The CPU spec for the MXFP4 training arena — layout resolution, staging into MXFP4-declared buffers, numerics against an oracle decoded from the source bytes — is `tests/test_mxfp4_arena_train.py`.

## Supported scope

- Families: gpt-oss (per-expert biases, clamped GLU, interleaved rows) and DeepSeek-V4 Flash / Pro (clamped SwiGLU, FP8 dense side).
- Arena provenance: a relocation bake is hash-preserving, so the served bytes are the checkpoint's own; a quantize-at-bake NF4 arena is bit-identical to the quantiser's output, not to the release. The manifest records `bake_mode`.
- Version floors, all from `pyproject.toml`: the `fast` extra's floor is grouped-nf4-gemm >= 0.30.0 at this commit (validated by CI). Two lower feature floors are recorded in its comment ladder and sit below the current floor: training on an MXFP4 arena landed in grouped-nf4-gemm 0.12.0 (the `F8_E8M0` scale tag; below it `check_arena_geometry` raises `KeyError`), and the MXFP4 grouped kernels' 64-bit expert offset (grouped-nf4-gemm#205) in 0.14.0.
- Environment: Linux, NVIDIA CUDA sm_80 or newer, Triton (Linux-only); CI tests Python 3.11.

## Limitations

- A uniform int4 grid cannot serve MXFP4 experts. For gpt-oss, `enable_serve_experts_int4` never re-quantises onto the int4 grid; it installs the **native MXFP4 store** served through `mxfp4_grouped`'s decode GEMV. That store is opt-in and single-stream-oriented (batched rows fall back to NF4 when the stacks are kept via `E4B_INT4_KEEP_NF4=1`), and its speed is quoted with the **quality gate open**: gpt-oss raw-text perplexity cannot rank an exact arm against a noisy one, and the pre-registered KL gate is falsified ([`../STATUS.md`](../STATUS.md)).
- `enable_fast` skips MXFP4-arena modules on purpose; their forward is wired by `nvme_experts`.
- `enable_mxfp4_nvme_residency` refuses `ExpertsLoRA`-wrapped modules: under the arena loader the base buffers are on `meta`, and binding would discard the adapter. Serve from the arena or train against it, not both on one load.
- Trainable LoRA over gpt-oss's biased, clamped experts needs a gpt-oss-aware adapter; that is a separate change (`arch/gptoss.py`).
- DeepSeek-V4's full-width resident load does not fit a small card; use the arena path.
- No shipped tool bakes a training arena from a bf16 checkpoint (open, `e4b.open.tr2-repro-gap`).
- Training on an MXFP4 arena has a CPU spec and a bench directory (`bench/mxfp4-arena-train/`) but no entry in the claims register: a capability, not a measured result.

## Related

- [`offload-moe-experts-to-cpu-or-nvme.md`](offload-moe-experts-to-cpu-or-nvme.md) · [`serve-large-moe-on-a-consumer-gpu.md`](serve-large-moe-on-a-consumer-gpu.md) · [`bitsandbytes-moe-load-in-4bit-still-ooms.md`](bitsandbytes-moe-load-in-4bit-still-ooms.md) · [`qlora-fused-moe-experts.md`](qlora-fused-moe-experts.md)
- [`../DEEPSEEK-V4.md`](../DEEPSEEK-V4.md) · [`../RESIDENCY-ENGINES.md`](../RESIDENCY-ENGINES.md) · [`../SERVING-THROUGHPUT.md`](../SERVING-THROUGHPUT.md) · [`../STATUS.md`](../STATUS.md)

## Evidence

Register: [`../claims.json`](../claims.json).

- `e4b.serve.gptoss.loader-faithful` — measured; the numeric receipt is in a private audit tree (`evidence_private`), the probe script is in-repo: MXFP4 dequant bit-identical to the reference, served expert path faithful to HF's reference math.
- `e4b.serve.deepseek-v4` — measured: DeepSeek-V4-Flash loads and generates with its experts served from an on-disk arena.
- `e4b.serve.informed-hot-sets` — measured: on V4-Flash's arena, profile-ranked hot sets beat by-index at identical VRAM.
- `e4b.serve.buildout.gptoss.b1.5090.2026-09-04` — measured: gpt-oss-20b's licensed stack is NF4 experts plus folds; the native MXFP4 store's row is speed with the quality gate open.
- Training on an MXFP4 arena: this page describes a capability; it carries no performance claim.
