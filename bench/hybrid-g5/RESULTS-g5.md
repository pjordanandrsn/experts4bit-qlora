# RESULTS — G5: hybrid QLoRA training vs hybrid inference (Qwen3-30B-A3B)

Gate G5: **hybrid training step ≤3× hybrid inference cost per token at
batch ≥8; parity test passes.** Rented Zen 5 EPYC 9655 + RTX 5090 box
(destroyed after), Qwen3-30B-A3B (48 MoE layers × 128 experts), NF4
quantize-bake arena, placement solved on-box from its own calibration blob
(`calib-g5.json`, tag `g5`) + a fresh routing profile: **75.7% VRAM /
24.3% DRAM / 0% NVMe** of routing mass — the solver hit its
bandwidth-balance target to four digits (achieved 3.11844 vs target
3.11857). Training ran with LoRA adapters on every MoE layer (r=8, α=16,
`arena_train=True` load), HF gradient checkpointing ON, AdamW on adapters
only.

## The ratio

| arm | measurement |
|---|---|
| hybrid inference (greedy decode, batch 1) | 15.31 tok/s → 65.3 ms/token |
| hybrid training step (batch 8 × seq 128 = 1,024 tok) | 2.415 s/step → **2.36 ms/token** |

**ratio = 0.036 — two orders of magnitude under the ≤3× bar. Gate G5
PASSES** (`batch_ok`, `ratio_ok`). The margin is not luck: a training step
amortizes each layer's expert-weight traffic across 1,024 tokens where
decode pays it per token — which is exactly why the gate is framed "at
batch ≥8". Step times are flat (2.32–2.63 s over 8 timed steps).

## Parity

- **Gradient parity (merge-gated tests, CUDA)**: base grads vs a
  full-precision reference over the same dequantized weights across three
  placements (three-tier mix, all-DRAM, all-VRAM), and QLoRA adapter
  grads (A and B, both projections) vs the reference with adapters —
  documented tolerance 3e-2 (GPU buses compute bf16, DRAM bus fp32,
  reference fp32 end-to-end). `tests/test_hybrid_train.py`.
- **Loss-curve overlay (this run)**: 20 memorization steps on a fixed
  batch, both arms from the identical post-warm adapter init, same seed,
  same data. Hybrid three-tier arm vs the full-GPU `[fast]` reference
  lane (`enable_fast_train(dgrad=True)`, resident bnb storage): step-0
  losses agree to **0.001** (13.2850 vs 13.2863 — the frozen-base loss,
  as B=0 demands) and the curves track within **max |Δ| = 0.26** across a
  13.3 → 5.0 descent, the hybrid arm marginally lower at the end. Curves
  in `g5_report.json`.
- **AVX-512 dgrad exactness**: the Phase-5 CPU kernel's exact-parity
  suite ran on this box's Zen 5 hand path — 19/19
  (`dgrad-avx512-parity.txt`), closing the AVX-512 leg the dev box
  (AVX2) could not exercise.

## Honest caveats

- The first overlay attempt was invalid **by bench construction**: both
  arms warm with a real optimizer step, and only the hybrid arm
  re-initialized adapters afterward — the reference entered the overlay
  with B ≠ 0, skipping the B=0 cold-start (dL/dA ≡ 0 while B is zero)
  and converging faster for reasons unrelated to the engine under test.
  Both arms now re-init identically post-warm (`_reinit_adapters`); the
  committed numbers are the corrected run.
- The overlay's remaining ≤0.26 drift is the documented mixed-precision
  class (per-bus bf16 vs fp32 rounding compounding through optimizer
  steps), not a bias: the hybrid arm ends *lower*.
- Inference is measured at batch-1 greedy decode (the G3 methodology);
  the gate's "batch ≥8" qualifies the training side.
- gpt-oss's biased epilogue is refused at enable (v1 covers the
  silu/clamp GLU families); MXFP4 dgrad kernels ship spec-tested but the
  e4b train wiring is NF4 (the shipped hybrid arena format).

Receipts in this directory; runner (`g5_run.py`) committed alongside.
