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

Claim family `e4b.train.parity.tp1.*` (measured, receipts in [`../../bench/train-parity-20260905/tp1/`](../../bench/train-parity-20260905/tp1/README.md); lane tp1, 2026-09-05, the shipped code on one rented RTX 5090, real weights through the direct loader, 60 steps on the registered `clinical` text, verdicts in the same registered units with every row classified OK / REFUSED / HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL, cost reported and never gated): `enable_fast_train(dgrad=True)` **PASSES** on OLMoE-1B-7B-Instruct (`e4b.train.parity.tp1.olmoe.fused.2026-09-05`, the first fused-vs-reference reading on a registered text with real weights for that family), on Qwen3-30B-A3B resident on the 32 GB card (`…qwen3.fused…`), on Gemma-4-26B-A4B-it — the `-it` checkpoint, loaded without #344 on that host — with the step-wise median inside the band by a small margin (`…gemma4.fused…`), and on Mixtral-8x7B-Instruct under `offload=True` at half the reference loop's peak VRAM (`…mixtral.fused…`; the family entered `model_families` on it); and on Granite-3.1-3B-A800M — the family's first direct real-weight load — on the corrected-counter re-run (`…granite.fused…`; the first attempt is a kept HARNESS_ERROR row, `…granite.fused.attempt1…`, a closure bug in the harness's kernel counter, not the shipped code; the family entered `model_families` on the re-run); `enable_batched_train` **PASSES** on Granite (`…granite.batched…`) and Mixtral (`…mixtral.batched…`) and is **VOID** on OLMoE, Qwen3 and Gemma-4 — the kernel was not reached on every layer (below); gpt-oss's fused and batched arms are **REFUSED** (below). A PASS there is a PASS on one text; no convergence claim and no cross-family ratio is made.

Claim family `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05` (measured, receipts in [`../../bench/h2h-20260905/p38/`](../../bench/h2h-20260905/p38/README.md); lane p38, 2026-09-05, one rented RTX 5090, the pre-registration verbatim in the bundle): the **head-to-head against Unsloth**, **end-to-end**, on one identical training problem — Qwen3-30B-A3B at one pinned revision, the registered `clinical` fixture tokenised once and asserted by sha in every arm, seq 512, r 8 / α 16 on attention q/k/v/o and every expert, the router frozen, the same optimizer, LR, batch, steps, precision and held-out eval on both sides; this package's fused `dgrad` path with NF4 attention (`enable_fast_train(dgrad=True)` + the shipped `TRAIN_ATTN_4BIT` mechanism) against Unsloth 2026.9.2's 4-bit MoE QLoRA path. At 60 steps e4b is faster per step (s/step ratio Unsloth/e4b 1.413: 2.151 vs 1.522 s), lower in peak VRAM (21.371 vs 23.141 GB) and energy per step (157.1 vs 224.7 J), reaches a held-out loss of 0.32 sooner (92.5 vs 130.3 s), and the two are comparable on held-out loss (0.2923 vs 0.2975, |Δ| 0.0052 ≤ the 0.05 reading threshold; `…quality-n60`). **At 200 steps the curves separate in Unsloth's favour** (0.2713 vs 0.2881, `…curve-n200`) — e4b's flattens near 0.29 from step 60 while Unsloth's keeps falling; that row is quoted beside the position wherever the position is quoted, and its causes (the eval schedule, the checkpointing mode, the two stacks' transformers/peft versions, the expert adapter's precision — bf16 here because the loader passes the model dtype to `ExpertsLoRA`, fp32 on Unsloth's side) are candidates, not established. One workload (≈86 tokens per step, batch 1, resident), one box, one family; no general speed claim; nothing licensed (`…e4b-internal-parity` is the informational fused-vs-reference PASS on that box, tp1 owns the licence).

## Symptoms

- "QLoRA a Qwen3 MoE on a 24 GB GPU" / "fine-tune fused MoE experts with LoRA on a consumer GPU".
- "experts4bit-qlora vs Unsloth for MoE QLoRA" / "is Unsloth faster than e4b on a 4-bit MoE" — the head-to-head above, one identical problem on one box.
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

- Families, per path and from evidence — the capability's `model_families` (`olmoe`, `qwen3_moe`, `gemma4_text`, `mixtral`, `granitemoe`) is exactly the families whose `fast_train` cell below is `supported`:

| model_type | quantize | reference_train | fast_train (the headline path) | batched_train | nvme_train | native_mxfp4_train |
|---|---|---|---|---|---|---|
| `olmoe` | supported (tp1) | supported (tp1; `e4b.train.olmoe-converges`) | **supported** — tp1 OK · PASS on the registered text with real weights (`e4b.train.parity.tp1.olmoe.fused.2026-09-05`) | **void** — tp1 OK · VOID: the `_PAD_WASTE_LIMIT` fallback engaged without a counter (`…olmoe.batched…`) | not_tested (the arena ladder is measured-private, no shipped bake) | n/a |
| `qwen3_moe` | supported (tp1, resident on a 32 GB card; flagship) | supported (tp1; flagship) | **supported** — tp1 OK · PASS resident on one 5090 (`…qwen3.fused…`), beside the flagship's five datasets (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: the kernel reached on a fraction of the layers every step (`…qwen3.batched…`); the dgrad-gate trajectory stands on its own fixture | not_tested (measured-private) | n/a |
| `gemma4_text` | supported (tp1: the `-it` checkpoint loaded on this host, no #344; flagship: base) | supported (tp1; flagship) | **supported** — tp1 OK · PASS with the step-wise median inside the band by a small margin (`…gemma4.fused…`), beside the model-2 flagship (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: on some steps no layer reached the kernel (`…gemma4.batched…`) | not_tested (measured-private) | n/a |
| `granitemoe` | supported (tp1: the first direct real-weight load) | supported (tp1 OK) | **supported** — tp1 OK · PASS on the corrected-counter re-run (`…granite.fused…`; attempt 1 a kept HARNESS_ERROR of the harness's counter, `…granite.fused.attempt1…`, amendment 3) — **entered `model_families` on it** | supported — tp1 OK · PASS (`…granite.batched…`) | not_tested | n/a |
| `gpt_oss` | supported (bare `GptOssExperts4bit`; tp1) | refused — no `ExpertsLoRA`; attention-only QLoRA trains (`…gptoss.attn_only…`, OK · no pair) | refused — `enable_fast_train` returns 0 (`…gptoss.fused…`, REFUSED) | refused — `enable_batched_train` returns 0 (`…gptoss.batched…`, REFUSED) | refused — `enable_mxfp4_nvme_residency` refuses bias-carrying modules (#402; it had defaulted to the V4 epilogue, #397), `enable_nvme_train_residency` refuses bare modules, and the `arena_train=True` wrap is refused on structure | **experimental** — grouped-nf4-gemm's `ExpertsMxfp4LoRA`; tp1 canary and provenance passed on its own text (`…gptoss.mxfp4…`, EXPERIMENTAL); never licensed |
| `mixtral` | supported (tp1: the first real-weight pass through the `w1/w3/w2` fusion, `offload=True`) | supported (tp1, offload) | **supported** — tp1 OK · PASS under offload at half the reference loop's peak VRAM (`…mixtral.fused…`); **entered `model_families` on this row** | supported — tp1 OK · PASS, the kernel reached everywhere (the 8-expert shape; `…mixtral.batched…`) | not_tested | n/a |

  Each cell is one of `supported` (completed under the registered protocol with a PASS/OK receipt), `refused` (with the reason), `void` (ran, unreadable), `harness_error`, `not_tested`, `experimental`, `n/a` — per path, never a flat flag; the machine-readable form, with the claim id behind every `supported` / `void` / `refused` cell, is `training_support` in [`../capabilities.json`](../capabilities.json), validated by `scripts/check_capabilities.py`, and `model_families` is exactly the families whose `fast_train` is `supported`. Row statuses in the tp1 receipt are one of OK / REFUSED / HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL with the parity verdict (PASS / FAIL / VOID) as a separate column. DeepSeek-V4 is admitted by structure (the base supplies its clamped epilogue via `_apply_gate`, [`../DEEPSEEK-V4.md`](../DEEPSEEK-V4.md)) and has no training receipt. The per-family training table with every row's status is the tp1 section of [`../ARCHITECTURE_SUPPORT.md`](../ARCHITECTURE_SUPPORT.md); the loader's families are in [`bitsandbytes-moe-load-in-4bit-still-ooms.md`](bitsandbytes-moe-load-in-4bit-still-ooms.md).
- Storage: nf4 is the benchmarked QLoRA default; fp4 / int8 / fp8 / bf16 / fp16 carry a tested LoRA-step contract only ([`../STORAGE-MODES.md`](../STORAGE-MODES.md)).
- `OFFLOAD_EXPERTS=1` / `offload=True` is what makes a 30B-class MoE train on a 12 GB or 24 GB card ([`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md)).
- Environment: Linux, NVIDIA CUDA, torch>=2.2, bitsandbytes>=0.43, transformers>=5.0; CI tests Python 3.11. `[fast]` installs grouped-nf4-gemm at the `fast` extra's floor in `pyproject.toml` (grouped-nf4-gemm >= 0.30.0 at this commit; validated by CI), which needs Triton (Linux-only) on an sm_80-or-newer GPU.

## Limitations

- `enable_fast_train` returns `0` rather than raising when `nf4_qlora` is missing; `dgrad=True` on a kernel cut older than grouped-nf4-gemm 0.7.0 is turned off with a `RuntimeWarning`. Assert the count.
- `train.main()` does not enable the fast path for you ([`../../CONTRIBUTING.md`](../../CONTRIBUTING.md)).
- The fused path changes the expert summation order (opt-in on purpose) and its model-level accuracy is measured, not assumed (`engines/fast.py` docstring).
- `enable_batched_train` wins only a toy microbench; at Qwen3-30B width it gives no speed-up and the highest peak memory (`engines/batched.py`, `e4b.train.fast-train-dgrad`) — width-dependent: on tp1 it is the faster of the two paths measured at Granite-3B width (`e4b.train.parity.tp1.granite.batched.2026-09-05`). It and `enable_fast_train` are mutually exclusive on a module.
- **What to check on a batched arm — engagement, not the count.** `enable_batched_train` falls back to the reference forward per call whenever the padded gather exceeds `_PAD_WASTE_LIMIT` (`engines/batched.py`), so its return value stays positive while some layers run the per-expert loop on some steps. In the code tp1 measured (0.35.0) that fallback was silent and uncounted, and three batched arms are **VOID** on exactly this — OLMoE, Qwen3 and Gemma-4 (`e4b.train.parity.tp1.olmoe.batched.2026-09-05`, `…qwen3.batched…`, `…gemma4.batched…`; on 9 of Gemma-4's 60 steps no layer reached the kernel) — while Mixtral's 8-expert and Granite's 40-expert shapes engaged everywhere. As of 0.35.1 the fallback is counted: read `batched_fallback_stats(model)` after a step (#402; those VOID rows are why it exists) and require at least two kernel calls per patched layer per step; a VOID row carries no parity number, however its loss curve reads.
- **A wrapper whose base breaks the stock-epilogue contract is refused, loudly.** `enable_fast`, `enable_fast_train` and `enable_batched_train` raise `EpilogueContractError` (0.35.1, #402) rather than compute `act(gate) * up` over a base that applies biases, clamps or another gating rule; DeepSeek-V4 supplies its epilogue through `_apply_gate` and is admitted, gpt-oss is the family the refusal protects.
- **What to check on gpt-oss — the count is zero, by design.** The loader builds its experts bare (no `ExpertsLoRA`), so `enable_fast_train` and `enable_batched_train` both return `0` — REFUSED rows on tp1 (`e4b.train.parity.tp1.gptoss.fused.2026-09-05`, `.batched`). Attention-only QLoRA over the frozen experts trains (`e4b.train.parity.tp1.gptoss.attn_only.2026-09-05`); the only route that trains its experts is the kernel package's experimental `ExpertsMxfp4LoRA` ([`mxfp4-moe-training-and-residency.md`](mxfp4-moe-training-and-residency.md)), never licensed.
- Non-checkpointed offload training is unsupported and fails loudly; the trainer always enables gradient checkpointing.
- gpt-oss: trainable LoRA over its biased, clamped experts needs a gpt-oss-aware adapter, which is a separate change (`arch/gptoss.py`).
- The registered training speed-up claim is retired: restated against a current baseline, roughly half of it is upstream's own fused loop (`e4b.retired.13.47x-training-speedup`, retired; [`../STATUS.md`](../STATUS.md), "What changed").
- **The Unsloth head-to-head is one workload on one box, and its 200-step curve favours Unsloth.** The 60-step position (e4b faster per step, lower peak VRAM and energy, comparable held-out loss) is never quoted without the 200-step row (`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05.curve-n200`: Unsloth's held-out loss lower at the end of the curve). ≈86 tokens per step at batch 1 on a resident 30B MoE; larger batches, longer sequences and other families are not measured; the candidate causes of the separation are not established.

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
- `e4b.train.parity.tp1.olmoe.fused.2026-09-05`, `e4b.train.parity.tp1.qwen3.fused.2026-09-05`, `e4b.train.parity.tp1.gemma4.fused.2026-09-05`, `e4b.train.parity.tp1.mixtral.fused.2026-09-05`, `e4b.train.parity.tp1.granite.fused.2026-09-05`, `e4b.train.parity.tp1.granite.batched.2026-09-05`, `e4b.train.parity.tp1.mixtral.batched.2026-09-05` — tp1 (real weights, the shipped code, one RTX 5090): the fused path passes on every family that has one and the batched path on Granite and Mixtral, in the registered units; `e4b.train.parity.tp1.<family>.matrix.2026-09-05` per family (load + verify + every arm's row, including the VOID and the refusals); every row of the lane is measured.
- `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05` — the head-to-head against Unsloth's 4-bit MoE QLoRA path, end-to-end, on one identical training problem (one rented RTX 5090): the s/step ratio, peak VRAM, energy and time-to-target at 60 steps; `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05.quality-n60` — held-out loss comparable at 60 steps; `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05.curve-n200` — Unsloth's held-out loss lower at 200 steps, quoted beside the position; `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05.e4b-internal-parity` — the fused-vs-reference PASS on that box (informational); one row per arm under `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05.arm.*`.
