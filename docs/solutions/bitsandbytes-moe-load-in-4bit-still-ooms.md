# Why does `load_in_4bit` still OOM on a Mixture-of-Experts model?

Because bitsandbytes' 4-bit walker only replaces `nn.Linear`, and transformers v5 stores a MoE's experts as one fused 3-D `nn.Parameter` per layer, so the experts — most of the weights — are silently left in bf16. `experts4bit-qlora` quantises exactly that fused stack on the way to the GPU, and `verify_moe_4bit(model, strict=True)` proves it happened.

## Symptoms

- `from_pretrained(..., load_in_4bit=True)` or a `BitsAndBytesConfig` "worked", but the model still OOMs, or `nvidia-smi` shows a near-bf16 footprint.
- `model.layers.N.mlp.experts.gate_up_proj` / `down_proj` are still `torch.bfloat16` after loading; only attention and the router got `Linear4bit`.
- "4-bit loading skips MoE experts" / "bitsandbytes does not quantize the fused 3-D expert tensor" / "the experts are not an `nn.Linear`".
- PEFT cannot target the experts either: `target_modules` finds no Linear there.
- You want to *check* whether the experts are actually 4-bit (NF4) rather than trust the config.

## Why it happens

transformers v5 fuses each MoE layer's experts into two stacked tensors, `gate_up_proj [E, 2I, H]` and `down_proj [E, H, I]`. the bitsandbytes 4-bit walker that `from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))` runs looks for `nn.Linear` modules; a 3-D parameter is not one, so it is skipped without a warning ([bitsandbytes#1849](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849)). The dense side shrinks, the experts do not, and the experts are the overwhelming majority of a MoE's bytes. Some checkpoints are worse: a naive `from_pretrained` of an MXFP4 release such as DeepSeek-V4 materialises the fp4 experts to bf16 first ([`../DEEPSEEK-V4.md`](../DEEPSEEK-V4.md)).

## Which project solves it

**experts4bit-qlora** owns loading and quantisation orchestration. Its `Experts4bit` primitive (the 4-bit face of `ExpertsNbit`; nf4 / fp4 / int8 / fp8 / bf16 / fp16 storage) is frozen quantised storage for a fused expert stack, with quantisation blocks that never cross an expert boundary. `load_moe_4bit_streaming` streams the checkpoint tensor-by-tensor onto the GPU, quantises each expert stack as it arrives and drops the bf16 source, so the bf16 model is never materialised in CPU or GPU RAM. `verify_moe_4bit` is the read-only check that works on any model, including one loaded the stock way. No kernel package is needed for this step; `grouped-nf4-gemm` only enters when you want the fused GEMM ([`qlora-fused-moe-experts.md`](qlora-fused-moe-experts.md)). Relationship to bitsandbytes and the upstream PR: [`../BITSANDBYTES.md`](../BITSANDBYTES.md).

## Install

```bash
pip install "experts4bit-qlora[train]"   # the streaming loader needs transformers>=5.0
```

`e4b`, `e4b-qlora`, `experts4bit`, `expertsnbit` and `experts-mxfp4` are lookup aliases of the same package; install the canonical name.

## Smallest correct example

Needs: GPU + network + model download.

```python
import torch
from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit

model, config = load_moe_4bit_streaming(
    "Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
)
report = verify_moe_4bit(model, strict=True)   # raises if any expert stack is still bf16
print(report["n_quantized"], report["n_unquantized"])
```

Needs: CPU-only. The same check on a model you loaded some other way:

```python
from experts4bit_qlora import verify_moe_4bit

report = verify_moe_4bit(stock_model)          # e.g. from_pretrained(..., load_in_4bit=True)
for stack in report["unquantized"]:
    print(stack["module"], stack["dtype"], stack["shape"])   # the stacks bitsandbytes skipped
```

## Expected result

`verify_moe_4bit` returns `{"quantized": [...], "unquantized": [...], "n_quantized": int, "n_unquantized": int}`. After the streaming loader, `n_unquantized == 0` and every `quantized` entry reports `quant_type == "nf4"` (or the `quant_type` you passed). On a stock 4-bit load, `unquantized` lists each `...experts.gate_up_proj` with its bf16 dtype and 3-D shape, and `strict=True` raises `RuntimeError` naming the count and the fix. The loader itself refuses to return a model on which it quantised zero expert layers.

## Supported scope

- Families (README "Scope", [`../ARCHITECTURE_SUPPORT.md`](../ARCHITECTURE_SUPPORT.md)): OLMoE, Qwen3-MoE / Qwen3.5-MoE, Gemma-4 (text tower), GraniteMoe, gpt-oss (MXFP4 experts, dequantised bit-identically), DeepSeek-V4 Flash / Pro; loaded with real weights in the support matrix: `olmoe`, `qwen3_moe`, `deepseek_v2`, `qwen3_next`. Mixtral-convention checkpoints are admitted through `arch/moe_conventions.py`.
- Storage: `quant_type=` selects nf4 / fp4 / int8 / fp8 / bf16 / fp16 ([`../STORAGE-MODES.md`](../STORAGE-MODES.md)); nf4 is the benchmarked default.
- Environment: Linux, an NVIDIA CUDA GPU, torch>=2.2, bitsandbytes>=0.43, transformers>=5.0. CI tests Python 3.11; `requires-python` says >=3.9 but older interpreters are not tested.

## Limitations

- An unsupported `model_type` fails fast with a clear error; LongCat-Flash's identity experts are refused by name; `deepseek_v3` is blocked by stale remote code in the tiny checkpoint ([`../ARCHITECTURE_SUPPORT.md`](../ARCHITECTURE_SUPPORT.md)).
- Detection in `verify_moe_4bit` is a heuristic: a module whose class name contains `Experts` holding a 3-D float parameter. A new family may need its class recognised.
- 4-bit on a card that already fits the model is a memory trade, not a speed-up, and costs energy: claim `e4b.train.energy-honest`.
- DeepSeek-V4's full-width resident load stacks one layer's experts in bf16 before quantising; use the arena path on a small card ([`../DEEPSEEK-V4.md`](../DEEPSEEK-V4.md)).
- Gemma-4-26B-A4B fails to load on some rented hosts after the experts quantise — open, [#344](https://github.com/pjordanandrsn/experts4bit-qlora/issues/344).
- `python -m experts4bit_qlora.verify --manifest ...` is the placement-manifest verifier, not the model check; the model check is the Python function above.

## Related

- [`qlora-fused-moe-experts.md`](qlora-fused-moe-experts.md) — train adapters over the quantised stack.
- [`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md) — when 4-bit experts still exceed VRAM.
- [`mxfp4-moe-training-and-residency.md`](mxfp4-moe-training-and-residency.md) — checkpoints released in MXFP4.
- [`../BITSANDBYTES.md`](../BITSANDBYTES.md) · [`../STORAGE-MODES.md`](../STORAGE-MODES.md) · [`../STATUS.md`](../STATUS.md) · [`../../README.md`](../../README.md)

## Evidence

Register: [`../claims.json`](../claims.json). Status words as in [`../STATUS.md`](../STATUS.md).

- `e4b.train.olmoe-fits` — measured: bf16 OLMoE-1B-7B OOMs a 12 GB card, 4-bit loads and trains; the loader never materialises bf16 under a container RAM cap.
- `e4b.serve.gptoss.loader-faithful` — measured; the numeric receipt is in a private audit tree (`evidence_private`), the probe script is in-repo: gpt-oss MXFP4 dequant is bit-identical to the reference.
- `e4b.train.energy-honest` — measured: storage-only 4-bit is an energy penalty when the model already fits.
