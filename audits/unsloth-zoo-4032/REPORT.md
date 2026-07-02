# Falsification report: unsloth-zoo's MoE bnb-4bit fix (unsloth#4032 close)

**Filed (2026-07-02):** Bug 1 → [unsloth-zoo#849](https://github.com/unslothai/unsloth-zoo/issues/849)
· Bug 2 → [unsloth-zoo#850](https://github.com/unslothai/unsloth-zoo/issues/850) · verification
comment on the original thread →
[unsloth#4032 (comment)](https://github.com/unslothai/unsloth/issues/4032#issuecomment-4870034310)

**Date:** 2026-07-02 · **Verdict: REAL-BUT-PARTIAL** — the quantization machinery is correct and
the headline model (Qwen3-30B-A3B) verifiably works end-to-end, but the audit found **two
previously-unreported bugs**: a silent wrong-math transposition for square-dimension experts on the
default grouped_mm backend (bf16 *and* 4-bit), and a quantize-without-routing crash that makes
OLMoE + `load_in_4bit` unusable.

## Environment

| component | version | note |
|---|---|---|
| unsloth | 2026.6.9 (PyPI latest, 2026-06-22) | |
| unsloth-zoo | 2026.6.7 (PyPI latest, 2026-06-22) | key files content-identical to GitHub main@2026-07-02 (modulo CRLF/LF line endings) |
| transformers | 5.5.0 | inside unsloth's declared `<=5.5.0` matrix |
| torch | 2.11.0+cu128 | **outside** unsloth's `<2.11` pin — disclosed; both bugs are pure shape/coverage logic, version-independent by code reading |
| bitsandbytes | 0.49.2 (release) | |
| GPU | RTX A2000 12GB (sm86) | torch 2.11 enables `torch._grouped_mm` here; on their pinned torch the same default path serves H100-class GPUs |
| install | `pip install --no-deps unsloth unsloth_zoo` overlaid (PYTHONPATH) on an existing torch/peft/accelerate stack; transformers/trl/datasets shadowed with in-matrix versions | |

## Verdict grid

| test | dims | quantized | routed to zoo bnb4bit dispatcher | parity vs stock math on their dequantized weights (default backend) | expert LoRA + grads |
|---|---|---|---|---|---|
| **Qwen3-30B-A3B** (real shard, layers 0–1) | non-square | ✓ 2/2 | ✓ | **✓** rel err 0.32 % = 0.65× bf16-noise control | ✓ 8 expert LoRA params, nonzero grads, step OK |
| **Gemma-4-26B-A4B** (real, 4 layers) | non-square | ✓ 4/4 | ✓ | **✓** 0.30 % = 0.96× control | adapters attach (16); train probe n/t on 10 GB free (262k-vocab CE OOM — harness limit) |
| **OLMoE-1B-7B-0924** (real, full) | gate_up **square** (2·1024 = 2048) | ✓ 16/16 (3.0 GiB) | **✗** — transformers' own experts kernel runs instead | **CRASH** — module-level: `got Byte`; full-model fwd: `IndexError` in transformers `_grouped_mm_fallback` (`weight.size(2)` on packed 2-D uint8) | adapters **attach** (64 expert LoRA params, both routes); training forward hits the same crash |
| Qwen3-MoE tiny (synthetic, square 2I=H=64), 4-bit | square | ✓ 2/2 | ✓ | **✗ SILENT**: rel err 109 %, **140.7× beyond noise**; loss finite, training "runs" | ✓ attaches — trains on wrong forward |
| Qwen3-MoE tiny, **bf16** (no quantization) | square | n/a | ✓ | **✗ SILENT**: 111 %, 139× — same bug without 4-bit | |
| tiny, `UNSLOTH_MOE_BACKEND=native_torch` | square | ✓ | ✓ | **✓ exact** (ratio 1.000 — max-abs error float-identical to bf16 control; outputs not elementwise-bitwise equal, differing reduction orders: `torch.equal` False) | |
| tiny, + transpose-always `preprocess_weight` | square | ✓ | ✓ | **✓ exact** (140.7 → 1.000, same max-abs-identical sense) | |
| `torch._grouped_mm` primitive sanity | — | — | — | exact (0.0 vs hand loop) | |
| OLMoE **bf16** (real, 4 layers) | square | 0/4 (as expected) | n/a (transformers kernel) | **✓** 0.87× control — bf16 OLMoE is fine | |

Parity method: module-level, same routing indices/weights; reference = fp32 loop over **their stored
quant_state dequantized with bnb's own `dequantize_4bit`** (never re-quantized); control = identical
math in bf16. "Excess over precision noise" = err(theirs vs fp32 ref) / err(bf16 ref vs fp32 ref).

## Bug 1 — silent transposition for square-dim experts (latent, wrong-math)

`unsloth_zoo/temporary_patches/moe_utils.py::preprocess_weight` converts transformers-v5 F.linear
layout (`gate_up (E, 2I, H)`, `down (E, H, I)`) to grouped_mm layout by **shape inspection**:
`if weight.shape[1] == hidden_dim: return weight` (gate_up). When `2I == H` (or `I == H` for down)
the check is ambiguous and returns the weight **untransposed** → every expert matmul runs on
transposed weights. Loss stays finite; nothing raises. The `_WEIGHT_PREPROCESSORS` registry that
could disambiguate per-arch has **zero registrations** (installed wheel and GitHub main).

- Repro: 8-line tiny Qwen3-MoE config with `2·moe_intermediate == hidden` (`make_tiny_qwen3.py` +
  `parity_only.py`). Affects bf16 and 4-bit equally; needs a grouped_mm-capable device.
- Proof of root cause: monkeypatching `preprocess_weight` to transpose-always drops excess from
  **140.7× to exactly 1.0** (`diag_rootcause.py`); `torch._grouped_mm` itself is exact.
- Blast radius: any F.linear-layout MoE arch with square dims routed through the zoo backend.
  Scanned shipping configs: official Qwen3-MoE / GLM-4.5 / ERNIE-4.5 / LFM2 / DeepSeek-V3 are
  non-square; **OLMoE is square** (blocked today only by Bug 2 — fixing Bug 2 the obvious way lands
  in Bug 1); **gpt-oss-20b/120b are down-square but SAFE** (transformers stores gpt-oss already in
  grouped layout, so no-transpose is correct there).
- Fix shape: layout is knowable from the source module class — register per-arch preprocessors (the
  registry exists, unused) or key on a `_unsloth_grouped_mm_format` attribute as their own PR #717
  did for GptOssExperts; don't shape-guess.
- Workaround: `UNSLOTH_MOE_BACKEND=native_torch` (exact parity, slower).

## Bug 2 — OLMoE: quantized by the generic matcher, never routed (shipping crash)

The quantizer patch (`moe_utils_bnb4bit.py`) matches **any** module with `gate_up_proj`/`down_proj`
nn.Parameters — OLMoE included (16/16 layers quantize, 3.0 GiB). But the forward routing that
dequantizes at compute time only covers the per-arch patch list (qwen3*, glm4, gemma4_moe,
deepseek_v3, lfm2, ernie4_5, gpt_oss…) — **no olmoe patch exists**, so transformers' own experts
kernel receives packed `(N,1)` uint8 storage and crashes on the first forward (archived repro
`results_olmoe_bare_forward.txt`:
`IndexError` at `transformers/integrations/moe.py::_grouped_mm_fallback` — `weight.size(2)` on the
2-D packed tensor; a module-level call on a grouped_mm device surfaces as `got Byte` instead — same
root cause). Quantize-coverage is generic; route-coverage is a list. Every fused-experts arch
outside the list = load fine, crash on first use. LoRA adapters attach fine on both routes; the
training forward then hits the same crash.

## What genuinely works (fairness)

- The #4032 core claim holds for its target: **real Qwen3-30B-A3B tensors quantize (NF4, correct
  bytes), route through the bnb4bit dispatcher, match stock math at noise floor, and train with
  expert adapters receiving nonzero grads.** Same for Gemma-4-26B-A4B (minus the train probe, which
  exceeded my 10 GB card for harness reasons).
- unsloth's `get_peft_model` with reporter-style `target_modules` **does attach expert adapters**
  via wrapped `experts.gate_up_proj`/`down_proj` params (no hollow pass).
- The quantization/dequant machinery is exact: `native_torch` backend parity is indistinguishable
  from the bf16-precision control everywhere tested (max-abs error float-identical; not claimed
  elementwise-bitwise — reduction orders differ, `torch.equal` is False).

## Mooting-race check (2026-07-02)

`preprocess_weight`, the empty registry, and the moe_bnb generic matcher are unchanged on
unsloth-zoo GitHub main; latest PyPI releases are 2026.6.9/2026.6.7 (June 22). Issue/PR search
found no existing report of either bug (nearest: zoo #717, a gpt-oss layout-flag fix — same bug
class, different instance; zoo #837 open PR for transformers<5 ModuleList experts).

## Artifacts (all in this directory, rerunnable)

`audit_lib.py` (audit/parity/train-probe harness) · `make_tiny_qwen3.py` (square repro model) ·
`run_falsify.py` (per-arch driver) · `parity_only.py` (backend/bf16 sweeps) · `diag_rootcause.py`
(primitive sanity + transpose-fix clincher + torch.equal check) ·
`results_{qwen3_tiny,olmoe,gemma4,qwen3_30b}.json` · `results_olmoe_bare_forward.txt` (bare
`model(ids)` IndexError trace) · `results_issue1_inline_repro.txt` (inline repro, both backends) ·
`results_olmoe_bf16.txt` (OLMoE bf16 truncated parity, 0.87× control)
