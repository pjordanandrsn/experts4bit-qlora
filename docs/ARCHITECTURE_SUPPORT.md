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

**Partial at this writing — the pending rows arrive before merge and this table is finalised with them.**

| model_type | checkpoint | direct load + `verify(strict)` | `ExpertsLoRA` | `fused` | `batched` | notes |
|---|---|---|---|---|---|---|
| `granitemoe` | granite-3.1-3b-a800m-instruct | validated — **real weights**, 32/32 layers, 2.35 GB (this file's `granitemoe` row above was a fixture) | 32 | pending (first run a harness error, re-run queued) | **PASS** (0.01553 / 0.01681; ×3.93 per step, J ×0.303) | enters the training capability when the fused re-run passes |
| `olmoe` | OLMoE-1B-7B-0924-Instruct | validated — **real weights**, 16/16, 4.70 GB | 16 | **PASS** (0.01327 / 0.01249; ×3.22, J ×0.339) | **VOID** — kernel reached on a minimum of 24 calls/step against 32 required: the `_PAD_WASTE_LIMIT` fallback engaged | first fused-vs-reference reading on a registered text with real weights |
| `gpt_oss` | gpt-oss-20b | validated — **real weights**, 24/24 as bare `GptOssExperts4bit`, 14.36 GB (this file's `gpt_oss` row above was `not_tested`) | 0 (built bare) | **REFUSED** (0 patched) | **REFUSED** (0 patched) | attention-only QLoRA over the frozen experts trains, stacks bit-exact (NO-PAIR); grouped-nf4-gemm's experimental `ExpertsMxfp4LoRA` route trains the experts on its own text, canary passing — experimental, not licensed |
| `qwen3_moe` | Qwen3-30B-A3B | validated — **real weights**, 48/48, **20.02 GB resident on a 32 GB card** | 48 | pending | pending | the reference arm was at step 20/60 at the snapshot |
| `gemma4_text` | gemma-4-26B-A4B-it | pending (#344: a load fault is a row) | pending | pending | pending | |
| `mixtral` | Mixtral-8x7B-Instruct-v0.1 | pending (`offload=True`; the first real-weight pass through the `w1/w3/w2` fusion) | pending | pending | pending | |

**Read the verdict columns as rows, not as a family score.** A VOID is not a PASS however its loss curve reads, and a
refusal is a row, not a zero. Nothing here is a convergence claim (60 steps rank two arms on one text), nothing is a
cross-family ratio, and the s/step figures in the bundle are this box's, quoted with its anchor class. The one code
finding the lane surfaced: `enable_batched_train` falls back to the reference forward per call above `_PAD_WASTE_LIMIT`
with no counter of its own, so a batched arm is read only with an external kernel-call counter.
