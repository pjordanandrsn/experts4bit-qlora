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
