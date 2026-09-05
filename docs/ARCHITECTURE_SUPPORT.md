# Architecture support and CUDA-graph capture

Companion to [support_matrix.md](support_matrix.md), which covers *(model, storage mode,
operation)*. This covers a different axis — **which architectures load, run, and capture** —
and comes from a different evidence bundle, so it is a separate file rather than an edit to
an OpenTimestamps-anchored document.

Same taxonomy as that file: `validated` reproduced under stated conditions · `broken`
expected to work, fails · `impractical` runs but exceeds a realistic budget · `unsupported`
not implemented · `not_tested` no evidence yet · `blocked` gated by hardware/access/library ·
`impossible` representation cannot be made safe.

## Evidence scope

- 2026-08-12. RTX 3090 (AMD EPYC 7452) and RTX A5000; torch 2.8.0, transformers 5.15.0,
  bitsandbytes 0.50.0, grouped-nf4-gemm 0.8.3.
- **Real-weight rows are marked.** Unmarked rows are synthetic 2-layer fixtures built at
  `hidden_size=128` (the public tiny-randoms are unusable here — `hidden_size` 8–32 is below
  the NF4 blocksize of 64, so the loader rejects them before any question is reached).
- Capture rows are 16 new tokens, greedy.

> **A fixture is not a checkpoint.** An earlier sweep of this exact matrix reported four
> families as failing. Re-run against **real published checkpoints**, three of the four
> loaded and ran — the failures were in configs I had generated, not in the loader. Treat any
> row below sourced from a synthetic fixture as evidence about the fixture until a real
> checkpoint agrees. The rows that changed are called out in the notes.

## Load + forward

| model_type | status | evidence |
|---|---|---|
| `olmoe` | validated | **real weights** — OLMoE-1B-7B-0924-Instruct |
| `qwen3_moe` | validated | **real weights** — Qwen3-30B-A3B, 48 layers, E=128, k=8, 18.7 GB VRAM |
| `deepseek_v2` | validated | **real weights** — `hmellor/tiny-random-DeepseekV2ForCausalLM`, 3 MoE layers, E=8 |
| `qwen3_next` | validated | **real weights** — `theo77186/Qwen3-Next-70M-TinyStories`, 8 MoE layers, E=16 |
| `ernie4_5_moe` | validated | fixture; needed the checkpoint-key renaming fix (see notes) |
| `qwen2_moe`, `granitemoe`, `hunyuan_v1_moe`, `glm4_moe`, `dots1` | validated | fixtures |
| `longcat_flash` | unsupported | identity ("zero-computation") experts — refused with the counts named |
| `deepseek_v3` | blocked | the tiny checkpoint's own remote code imports `is_torch_fx_available`, removed upstream. Not a loader defect; a checkpoint without stale remote code is `not_tested`. |
| `gpt_oss` | not_tested | the synthetic fixture's expert stacks were malformed; real checkpoints untested here |

## CUDA-graph decode capture

Capture requires the pipelined engine to be serving — the reference forward's dequantize path
synchronizes, and capture throws on a host sync inside the region.

| model_type | capture | speedup | notes |
|---|---|---|---|
| `olmoe` | validated | **1.11×** | **real weights**, 3090 — replays bit-identical to eager |
| `qwen3_moe` | validated | **1.04×** | **real weights**, Qwen3-30B-A3B on A5000 — bit-identical over 31 teacher-forced steps |
| `hunyuan_v1_moe` | validated | 5.44× | fixture |
| `qwen2_moe` | validated | 5.16× | fixture |
| `granitemoe` | validated | 4.98× | fixture |
| `glm4_moe` | validated | 4.72× | fixture |
| `dots1` | validated | 4.41× | fixture |
| `qwen3_next` | unsupported | — | `StaticCache` does not cover LinearAttention layers (`get_seq_length` raises). Loads and runs fine; only capture is unavailable. |

**Read the speedup column as a curve, not a number.** It is inversely proportional to how much
real GPU work a decode step does, which is what a *fixed* per-step launch cost predicts: ~5× on
2-layer fixtures that are nearly pure launch overhead, 1.11× at 1B active parameters, 1.04× at
30B. Capture is worth having for small models and for the zero-host-sync contract it enforces
(a successful capture proves no host sync occurs in the region). It is not a throughput result
at scale, and the fixture numbers should not be quoted as one.

## Notes

**`ernie4_5_moe`** — both released checkpoints (ERNIE-4.5-21B-A3B, 300B-A47B) store
`model.layers.N.mlp.moe_statics.e_score_correction_bias` at the MoE-block level — 27 and 51
keys respectively — while the module tree carries it under `mlp.gate.`. The loader walked raw
checkpoint keys and died with `Ernie4_5_MoeSparseMoeBlock has no attribute 'moe_statics'`. It
now reads transformers' own per-`model_type` rename table rather than keeping a second copy.

**`longcat_flash`** — allocates `gate_up_proj` over `n_routed_experts + zero_expert_num`
(512 + 256 by default) but `down_proj` over the routed count only, because its forward sends
`expert_idx >= num_routed_experts` through `nn.Identity` scaled by the router weight and never
reads those `gate_up` rows. Loading only the routed experts is not a fix: the router keeps
selecting over the full space. Supporting it needs an identity slot in the expert primitive.

**Rows that changed after re-running against real checkpoints:** `deepseek_v2` and `qwen3_next`
moved from *broken* to *validated*, and `deepseek_v3` from *broken* to *blocked*. Only
`ernie4_5_moe` survived as a real defect.

## Reproducing

The probe reports support rather than assuming it, and distinguishes a bf16 argmax tie from a
real defect by measurement — it teacher-forces the same tokens down both paths and compares
logits, against eager *incremental* decode (a full-sequence forward differs by a few ulp in
pure eager and would charge that to capture).

```python
from experts4bit_qlora import probe_capture
report = probe_capture(model, input_ids)   # captured / matches_eager / teacher_forced
```

## Training on real weights (tp1, 2026-09-05)

A different evidence bundle again — [`bench/train-parity-20260905/tp1/`](../bench/train-parity-20260905/tp1/README.md)
(lane tp1: the shipped experts4bit-qlora 0.35.0 + grouped-nf4-gemm 0.30.0, one rented RTX 5090 on an EPYC 7Q83 host,
train-anchor class `pcie-full/launch-fast`) — so it is a new dated section, not an edit to the tables above. Each of the
six serving families is loaded through the direct `load_moe_4bit_streaming` path on its **real** checkpoint (the
`-Instruct` / `-it` bytes the serving lanes use), `verify_moe_4bit(strict=True)` is run, and three training arms of 60
steps on the registered `clinical` text are compared in the registered units: `reference` (the per-expert loop), `fused`
(`enable_fast_train(model, dgrad=True)`), `batched` (`enable_batched_train(model)`). **PASS** = |Δ final train loss|
≤ 0.05 and median step-wise |Δ| ≤ 0.05 against the family's own reference arm, same box, same session, with `init_sha`
identical, the frozen stack bit-exact (C1) and the accelerated kernel reached on every layer; **VOID** = the row cannot
be read (the reason is stated); **REFUSED** = the enabler patched nothing. Cost is reported per row in the bundle and
never gated. Same taxonomy for the load column as the tables above; verdicts are the register's
(`e4b.train.parity.tp1.<family>.<arm>.2026-09-05` in [`claims.json`](claims.json)).

**All six families through the registered arms (`TP_DONE` 2026-09-05T15:22Z) and Granite's `fused` corrected-counter
re-run (`TP2_DONE` 15:33Z): 18 result lines, every attempt a row, nothing pending.**

| model_type | checkpoint | direct load + `verify(strict)` | `ExpertsLoRA` | `fused` | `batched` | notes |
|---|---|---|---|---|---|---|
| `granitemoe` | granite-3.1-3b-a800m-instruct | validated — **real weights**, 32/32 layers, 2.35 GB (this file's `granitemoe` row above was a fixture) | 32 | **PASS** on the corrected-counter re-run (0.01329 / 0.01270; ×5.86 per step, J ×0.197) — attempt 1 a kept **HARNESS_ERROR** (amendment 3; the harness's counter, not the shipped code) — **enters the training capability on it** | **PASS** (0.01553 / 0.01681; ×3.93 per step, J ×0.303) | both accelerated arms PASS; the family's first training receipt |
| `olmoe` | OLMoE-1B-7B-0924-Instruct | validated — **real weights**, 16/16, 4.70 GB | 16 | **PASS** (0.01327 / 0.01249; ×3.22, J ×0.339) | **VOID** — kernel reached on a minimum of 24 calls/step against 32 required: the `_PAD_WASTE_LIMIT` fallback engaged | first fused-vs-reference reading on a registered text with real weights |
| `gpt_oss` | gpt-oss-20b | validated — **real weights**, 24/24 as bare `GptOssExperts4bit`, 14.36 GB (this file's `gpt_oss` row above was `not_tested`) | 0 (built bare) | **REFUSED** (0 patched) | **REFUSED** (0 patched) | attention-only QLoRA over the frozen experts trains, stacks bit-exact (NO-PAIR); grouped-nf4-gemm's experimental `ExpertsMxfp4LoRA` route trains the experts on its own text, canary passing — experimental, not licensed |
| `qwen3_moe` | Qwen3-30B-A3B | validated — **real weights**, 48/48, **20.02 GB resident on a 32 GB card** | 48 | **PASS** (0.01315 / 0.01050; ×2.56 per step, J ×0.407) | **VOID** — the kernel reached on a fraction of the layers every step (min 12 calls/step against 96) | beside the flagship's five-dataset PASS (a 4090, offload), never divided into it |
| `gemma4_text` | gemma-4-26B-A4B-it | validated — **real weights**, the `-it` checkpoint, 30/30, 18.11 GB resident; **no #344 fault on this host** | 30 | **PASS** (0.02385 / 0.04742 — inside the band by 0.0026, the tightest cell; ×2.37, J ×0.477) | **VOID** — on 9 of 60 steps no layer reached the kernel (min 0 against 60) | the fetch hang of amendment 4 recovered inside its alarm; no ALARM row |
| `mixtral` | Mixtral-8x7B-Instruct-v0.1 | validated — **real weights**, 32/32 through the `w1/w3/w2` fusion, `offload=True` (3.42 GB on the GPU, 25.37 GB of experts pinned in host RAM) | 32 | **PASS** (0.00953 / 0.00945; ×1.26, peak ×0.494, J ×0.734) — **enters the training capability on this row** | **PASS** (0.00766 / 0.01057; ×1.27, peak ×0.802) — the 8-expert shape engaged the kernel everywhere | the family's first training receipt |

**Row statuses and per-path support (phase directive 2026-09-05).** Every row in the bundle's `RESULTS-tp1.md` is
exactly one of OK / REFUSED / HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL — classified by its reducer from
`summary.txt` (rc per attempt), the receipt or stub, `outer.log` and the run logs, never from a missing error — with
the parity verdict (PASS / FAIL / VOID) as a separate column for OK rows; Granite's first `fused` attempt is a
HARNESS_ERROR row kept beside its corrected-counter re-run, and every amendment is referenced from the rows it touched.
Read per path:

| model_type | quantize | reference_train | fast_train (the headline path) | batched_train | nvme_train | native_mxfp4_train |
|---|---|---|---|---|---|---|
| `olmoe` | supported (tp1) | supported (tp1; `e4b.train.olmoe-converges`) | **supported** — tp1 OK · PASS on the registered text with real weights (`e4b.train.parity.tp1.olmoe.fused.2026-09-05`) | **void** — tp1 OK · VOID: the `_PAD_WASTE_LIMIT` fallback engaged without a counter (`…olmoe.batched…`) | not_tested (the arena ladder is measured-private, no shipped bake) | n/a |
| `qwen3_moe` | supported (tp1, resident on a 32 GB card; flagship) | supported (tp1; flagship) | **supported** — tp1 OK · PASS resident on one 5090 (`…qwen3.fused…`), beside the flagship's five datasets (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: the kernel reached on a fraction of the layers every step (`…qwen3.batched…`); the dgrad-gate trajectory stands on its own fixture | not_tested (measured-private) | n/a |
| `gemma4_text` | supported (tp1: the `-it` checkpoint loaded on this host, no #344; flagship: base) | supported (tp1; flagship) | **supported** — tp1 OK · PASS with the step-wise median inside the band by a small margin (`…gemma4.fused…`), beside the model-2 flagship (`e4b.train.flagship-matrix`) | **void** — tp1 OK · VOID: on some steps no layer reached the kernel (`…gemma4.batched…`) | not_tested (measured-private) | n/a |
| `granitemoe` | supported (tp1: the first direct real-weight load) | supported (tp1 OK) | **supported** — tp1 OK · PASS on the corrected-counter re-run (`…granite.fused…`; attempt 1 a kept HARNESS_ERROR of the harness's counter, `…granite.fused.attempt1…`, amendment 3) — **entered `model_families` on it** | supported — tp1 OK · PASS (`…granite.batched…`) | not_tested | n/a |
| `gpt_oss` | supported (bare `GptOssExperts4bit`; tp1) | refused — no `ExpertsLoRA`; attention-only QLoRA trains (`…gptoss.attn_only…`, OK · no pair) | refused — `enable_fast_train` returns 0 (`…gptoss.fused…`, REFUSED) | refused — `enable_batched_train` returns 0 (`…gptoss.batched…`, REFUSED) | refused — `enable_mxfp4_nvme_residency` refuses bias-carrying modules (#402; it had defaulted to the V4 epilogue, #397), `enable_nvme_train_residency` refuses bare modules, and the `arena_train=True` wrap is refused on structure | **experimental** — grouped-nf4-gemm's `ExpertsMxfp4LoRA`; tp1 canary and provenance passed on its own text (`…gptoss.mxfp4…`, EXPERIMENTAL); never licensed |
| `mixtral` | supported (tp1: the first real-weight pass through the `w1/w3/w2` fusion, `offload=True`) | supported (tp1, offload) | **supported** — tp1 OK · PASS under offload at half the reference loop's peak VRAM (`…mixtral.fused…`); **entered `model_families` on this row** | supported — tp1 OK · PASS, the kernel reached everywhere (the 8-expert shape; `…mixtral.batched…`) | not_tested | n/a |

Each cell is one of `supported` (completed under the registered protocol with a PASS/OK receipt), `refused` (with the reason), `void` (ran, unreadable), `harness_error`, `not_tested`, `experimental`, `n/a` — per path, never a flat flag; the machine-readable form, with the claim id behind every `supported` / `void` / `refused` cell, is `training_support` in [`capabilities.json`](capabilities.json), validated by `scripts/check_capabilities.py`, and `model_families` is exactly the families whose `fast_train` is `supported`. Row statuses in the tp1 receipt are one of OK / REFUSED / HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL with the parity verdict (PASS / FAIL / VOID) as a separate column.

**Read the verdict columns as rows, not as a family score.** A VOID is not a PASS however its loss curve reads, and a
refusal is a row, not a zero. Nothing here is a convergence claim (60 steps rank two arms on one text), nothing is a
cross-family ratio, and the s/step figures in the bundle are this box's, quoted with its anchor class. The one code
finding the lane surfaced: in the code it measured (0.35.0) `enable_batched_train` falls back to the reference forward
per call above `_PAD_WASTE_LIMIT` with no counter of its own — three VOID rows (OLMoE, Qwen3, Gemma-4) against two
engaged ones (Granite, Mixtral) — so a batched arm was read only with an external kernel-call counter; 0.35.1 (#402)
adds `batched_fallback_stats(model)` for exactly this.
