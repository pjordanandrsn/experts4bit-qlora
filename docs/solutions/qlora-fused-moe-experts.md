# How do I QLoRA-train the fused experts of a MoE (per-expert LoRA on 4-bit experts)?
<!-- summary: ExpertsLoRA adds a trainable per-expert low-rank delta before each routed expert's activation over a frozen NF4 stack PEFT cannot target; enable_fast_train runs it on grouped kernels. -->

Load with `load_moe_4bit_streaming`, which installs a frozen NF4 `Experts4bit` stack wrapped in trainable per-expert `ExpertsLoRA` adapters for every MoE layer, then train with `python -m experts4bit_qlora.train` or your own loop. Add `enable_fast_train(model, dgrad=True)` from the `[fast]` extra to run the step through `grouped-nf4-gemm`'s grouped kernels, and assert its return value.

## The abstraction that fails, and what replaces it

PEFT expects `nn.Linear` targets. A fused MoE checkpoint stores each layer's experts as stacked 3-D projections — `gate_up_proj [E, 2I, H]` and `down_proj [E, H, I]` — so there is no per-expert `Linear` for `target_modules` to match. A delta hung on the stack as a whole would also compute the wrong function: the low-rank term has to be applied per routed expert *before* that expert's activation (`act(W_e x + B[e] A[e] x)`, not `act(W_e x) + d`). `ExpertsLoRA` supplies the adapter abstraction — stacked per-expert `A[e]` / `B[e]` over the frozen packed base, the delta injected pre-activation — and `grouped-nf4-gemm` supplies the fused grouped forward and, with `dgrad=True`, the optional input-gradient kernel.

```text
Ordinary PEFT:     nn.Linear per projection  <-  LoRA A/B hooked on each Linear
Fused checkpoint:  experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I]   (no per-expert Linear)
        |
        v
Experts4bit frozen packed base   +   ExpertsLoRA A[e] / B[e], one pair per expert
        |
        v
per-routed-expert delta, applied BEFORE the activation:  act(W_e x + B[e] A[e] x)
        |
        v
grouped forward (grouped-nf4-gemm)   +   optional dgrad (enable_fast_train(model, dgrad=True))
```

| component | state | where it is decided |
|---|---|---|
| packed expert base (`Experts4bit`, NF4 by default) | frozen | `quant_type=` on the loader; `ExpertsLoRA.__init__` sets `requires_grad=False` on every base parameter |
| per-expert `A[e]` / `B[e]` (`ExpertsLoRA`) | trainable; `B` is zero-initialised, so the delta starts at exactly zero | `r`, `alpha` on the loader; `R`, `ALPHA` in the trainer |
| router (`mlp.gate.weight`) | **frozen by default** — `TRAIN_ROUTER=0`; `TRAIN_ROUTER=1` trains it at 0.1× the LoRA learning rate | its own trainer switch, separate from `TRAIN_EXPERTS` and `TRAIN_ATTENTION`; the trainer selects parameter names ending in `mlp.gate.weight`, `ExpertsLoRA` itself never touches the router, and the placement ablation found training it hurts (`e4b.train.ablation`) |
| attention LoRA (`LoRALinear` over the frozen q/k/v/o) | trainable by default (`TRAIN_ATTENTION=1`) | a trainer switch |
| expert execution | reference per-expert loop (default), batched fallback (`enable_batched_train`), or fused grouped path (`enable_fast_train`, optional `dgrad=True`) | your call after loading; `train.main()` enables neither for you |
| base identity | verified, not assumed | `verify_moe_4bit(model, strict=True)` at load; the flagship gate verified the frozen 4-bit stack bit-identical over the hashed bytes (`e4b.train.flagship-matrix`); `python -m experts4bit_qlora.infer` serves the adapter over the same NF4 base it was trained against |

## Measured result

Claim `e4b.train.flagship-matrix` (measured, receipts in this repository): across two 30B-class MoEs — Qwen3-30B-A3B and Gemma-4-26B-A4B — five datasets each, 200 steps per cell, the fused training path runs 1.52–1.81× faster per step at 0.75–0.81× peak VRAM and 0.86–0.92× energy per step, with held-out loss parity on both registered criteria and the frozen 4-bit stack verified bit-identical over 16.31 GB hashed. That ratio is fused-vs-reference with the same offload and checkpointing in both arms. A per-step multiple quoted against the per-expert Python loop on a smaller model (the OLMoE per-expert-loop comparison) has a different denominator and different models; it is not this number, and the two do not compose.

## Symptoms

- "QLoRA a Qwen3 MoE on a 24 GB GPU" / "fine-tune fused MoE experts with LoRA on a consumer GPU".
- PEFT does not see the expert parameters: `target_modules` matches no `nn.Linear` under `mlp.experts`, because the experts are a fused 3-D tensor (`gate_up_proj`, `down_proj`).
- Stock `load_in_4bit` leaves the experts in bf16 and the run OOMs ([`bitsandbytes-moe-load-in-4bit-still-ooms.md`](bitsandbytes-moe-load-in-4bit-still-ooms.md)).
- Training works but every step crawls through a per-expert Python loop.
- `enable_fast_train` returned `0` (or something falsy) and nothing changed.
- You need loss parity with a bf16 run, not just a falling curve.

## Why it happens

Adapter libraries attach low-rank deltas to `nn.Linear`. A fused expert stack has no per-expert Linear to hook, and the adapter must land *before* the nonlinearity for each routed expert. `ExpertsLoRA` therefore re-implements the expert math: for each expert `e` the frozen 4-bit projections get a trainable `scaling * (x @ A[e].T) @ B[e].T` term, with stacked `A`/`B` per expert, while the base's projection re-dequantises in backward so no dequantised-expert activation is held (`experts4bit_qlora/lora.py`, [`../BITSANDBYTES.md`](../BITSANDBYTES.md)). The reference forward is a per-expert loop by design; the speed comes from the kernel side.

## Which project solves it

**experts4bit-qlora** owns the adapter (`ExpertsLoRA`), the streaming loader, the trainer and the training-side offload. **grouped-nf4-gemm** ([GitHub](https://github.com/pjordanandrsn/grouped-nf4-gemm), [PyPI](https://pypi.org/project/grouped-nf4-gemm/)) owns the kernels: `enable_fast_train` routes the `ExpertsLoRA` training forward through its `nf4_qlora.fused_grouped_lora`, and `dgrad=True` routes the backward through its single-launch dgrad kernel. `[fast]` is the seam. `enable_batched_train` is the no-extras fallback when the kernel package will not build.

## Install

```bash
pip install "experts4bit-qlora[train]"   # minimum/reference training: loader + trainer
pip install "experts4bit-qlora[fast]"    # accelerated grouped-kernel path: + grouped-nf4-gemm for enable_fast_train
```

## Smallest correct example

Needs: GPU + network + model download.

```python
import torch
from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming, verify_moe_4bit

model, config = load_moe_4bit_streaming(
    "Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16, r=8, alpha=16,
    quant_type="nf4", offload=True,          # offload=True: experts in pinned host RAM, one layer on the GPU
)
verify_moe_4bit(model, strict=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # required with offload
n = enable_fast_train(model, dgrad=True)
assert n > 0, "still on the per-expert loop: grouped-nf4-gemm missing?"
# ... your optimiser over the parameters whose name contains "lora" ...
```

The assertion is there because acceleration must be observed, not assumed: `enable_fast_train` returns `0` rather than raising when the kernel package is absent, and from the caller's side `0` looks exactly like the per-expert loop it was meant to replace. Keep the assert in your own loop.

The shipped trainer does the same from environment variables (there are no CLI flags; `--help` prints them):

```bash
STEPS=150 R=8 TRAIN_EXPERTS=1 OFFLOAD_EXPERTS=1 OUT=./out python -m experts4bit_qlora.train
```

## Expected result

`verify_moe_4bit(model, strict=True)` returns without raising. `enable_fast_train` returns the number of `ExpertsLoRA` modules patched — one per MoE layer — and `0` means you are silently on the reference loop ([`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) explains why every enabler returns a count). The trainer writes the adapter (`adapter_best.pt`) under `OUT`; `python -m experts4bit_qlora.infer` serves it over the same NF4 base with `ADAPTER=./out/adapter_best.pt`.

## Supported scope

- Families: the trainer docstring names OLMoE, Qwen3-MoE and Gemma-4; the loader's families are in [`bitsandbytes-moe-load-in-4bit-still-ooms.md`](bitsandbytes-moe-load-in-4bit-still-ooms.md). DeepSeek-V4 is trainable: the base supplies its clamped epilogue via `_apply_gate` ([`../DEEPSEEK-V4.md`](../DEEPSEEK-V4.md)).
- Storage: nf4 is the benchmarked QLoRA default; fp4 / int8 / fp8 / bf16 / fp16 carry a tested LoRA-step contract only ([`../STORAGE-MODES.md`](../STORAGE-MODES.md)).
- `OFFLOAD_EXPERTS=1` / `offload=True` is what makes a 30B-class MoE train on a 12 GB or 24 GB card ([`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md)).
- Environment: Linux, NVIDIA CUDA, torch>=2.2, bitsandbytes>=0.43, transformers>=5.0; CI tests Python 3.11. `[fast]` needs grouped-nf4-gemm>=0.28.0, which needs Triton (Linux-only) on an sm_80-or-newer GPU.

## Limitations

- `enable_fast_train` returns `0` rather than raising when `nf4_qlora` is missing; `dgrad=True` on a kernel cut older than grouped-nf4-gemm 0.7.0 is turned off with a `RuntimeWarning`. Assert the count.
- `train.main()` does not enable the fast path for you ([`../../CONTRIBUTING.md`](../../CONTRIBUTING.md)).
- The fused path changes the expert summation order (opt-in on purpose) and its model-level accuracy is measured, not assumed (`engines/fast.py` docstring).
- `enable_batched_train` wins only a toy microbench; at real width it gives no speed-up and the highest peak memory (`engines/batched.py`). It and `enable_fast_train` are mutually exclusive on a module.
- Non-checkpointed offload training is unsupported and fails loudly; the trainer always enables gradient checkpointing.
- gpt-oss: trainable LoRA over its biased, clamped experts needs a gpt-oss-aware adapter, which is a separate change (`arch/gptoss.py`).
- The registered training speed-up claim is superseded: restated against a current baseline, roughly half of it is upstream's own fused loop (`e4b.retired.13.47x-training-speedup`; [`../STATUS.md`](../STATUS.md), "What changed").

## Related

- [`bitsandbytes-moe-load-in-4bit-still-ooms.md`](bitsandbytes-moe-load-in-4bit-still-ooms.md) · [`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md) · [`offload-moe-experts-to-cpu-or-nvme.md`](offload-moe-experts-to-cpu-or-nvme.md) · [`mxfp4-moe-training-and-residency.md`](mxfp4-moe-training-and-residency.md)
- [`../CHOOSING.md`](../CHOOSING.md) · [`../METHODOLOGY.md`](../METHODOLOGY.md) · [`../STATUS.md`](../STATUS.md) · [`../../README.md`](../../README.md)

## Evidence

Register: [`../claims.json`](../claims.json); all of these are **measured** with receipts in this repository.

- `e4b.train.olmoe-converges` — QLoRA on frozen NF4 experts improves a held-out eval on OLMoE-1B-7B.
- `e4b.train.ablation` — adapter placement: experts+attention is the best eval, attention-only the efficiency pick; training the router hurts.
- `e4b.train.fast-train-dgrad` — `enable_fast_train(dgrad=True)` is the fastest lane at real width; the batched path wins only the microbench.
- `e4b.train.flagship-matrix` — two 30B-class MoEs by five datasets: fused path faster per step at lower VRAM and energy, loss parity on the registered criteria, frozen stack bit-identical.
- `e4b.offload.fits-30b-class` — with expert offload, Qwen3-30B-A3B and Gemma-4-26B-A4B QLoRA-train on 12 GB.
