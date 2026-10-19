# DQ11 MLP-adapter BUILD — registration (#1083)

**Status:** registered on merge, before any code or data. All six decisions below are ruled (maintainer, 2026-10-19,
before data). Merging costs $0. The engine is written only after this merges; Phase 1 runs only on the merged engine,
and Phase 2 runs only after `PHASE1_PASS`, its review and its own approval.

## Why

- **DQ11** (`dq11-5090-4f`; RESULTS-dq11.md) read `BUILD_CANDIDATE` and `U_FASTER_CONTROLLED` on Unsloth's fused
  QKV/O/MLP bundle. That read licenses "a separately reviewed fusion build investigation". It measured the whole bundle
  on Unsloth's resident backbone, not an MLP-only or e4b number.
- **The L-census** (`dq11-l-census-9`; two independent byte-identical reductions, #1083) read `BUILD_MLP`:
  - C_MLP = 0.1277 s ≥ 0.10·T = 0.0724 s, where T (the reference median) = 0.7245 s;
  - qkv 0.0225 s and o 0.0104 s, both below the line;
  - `decision.json` sha256 `66149541ed2971d0b824a9bf0084d43534ec11fbe6cdf1db03263d5a4432ae28`.
- **Overlap.** The MLP-adapter class's mean *exclusive* device time is 65.2 ms per update (≈ 0.090·T), against a
  union of 127.7 ms. About half of the union overlaps other classes, and overlapping work can only reduce realized
  savings (DQ11-L-CENSUS-PREREG.md).
- **The bar the census fixes:** "The prospective BUILD continuation bar remains 5% of an independently matched unfused
  update, subject to unchanged K8." No speed claim, default, capacity claim or launch follows from the census itself.

## The subject — Decision D1 (scope): RULED as proposed

A training engine in e4b, `enable_dense_mlp_lora_fuse(model)`:
- **Knob:** `E4B_DENSE_MLP_LORA_FUSE`, off by default. With it off, the path is bitwise today's PEFT path.
- **Targets:** the PEFT LoRA modules on each decoder layer's MLP (`gate_proj`, `up_proj`, `down_proj`; 96 adapters on
  Mistral-7B). Its count is returned, and an unexpected module is refused rather than skipped.
- **Refusals:** a module keeps today's PEFT path, with the reason recorded and counted, if any of these holds:
  - its LoRA dropout is not 0;
  - it uses DoRA;
  - more than one adapter is active;
  - the adapter is merged;
  - the adapter weights are not FP32;
  - its base is not a bitsandbytes NF4 `Linear4bit`, or it has a bias.
- **gate and up share their input:**
  - one FP32 cast of x;
  - one rank GEMM against the stacked `[A_gate; A_up]` (2r × 4096);
  - two B GEMMs, with the scale folded in.
- **One fused epilogue** adds each adapter's delta into the base output (`addmm_`-style), with one cast back.
- **Autograd:** a custom Function returns the same gradients (x, every A and B). Checkpoint recompute goes through the
  same Function.
- **Precision unchanged:** rank compute stays FP32. BF16 rank compute is a precision change, not this lever.

**Out of scope:**
- SwiGLU and gate×up (the census's "other" class);
- the base NF4 GEMM;
- the qkv and o adapters (below the census line);
- any MoE path.

**Not taken:**
- (b) epilogue-only, with no shared gate/up input;
- (c) an Unsloth-style LoRA_MLP that includes SwiGLU, which reaches past C_MLP into other classes.

## Fixture

- **Carried from the L science path, unchanged:**
  - Mistral-7B-v0.1 at `27d67f1b`;
  - seven targets, r16 / α32 / dropout 0, FP32 adapters;
  - NF4 bases with BF16 compute;
  - seq 2048, mb1/acc1, non-reentrant checkpointing, TF32 off;
  - streamed with `train_prefetch`, 40 AdamW updates, seed 3407.
- **New:** e4b at the merge head that carries the engine. The science closure records that head in place of
  `13b55990`.
- **Decision D2 (code location): RULED.** The engine lives in e4b (`experts4bit_qlora/engines/dense_lora_fuse.py`) and
  patches only the PEFT `LoraLayer` forward of the MLP modules. P129's `train_qkv_fuse` is its template. Not Loggetta:
  the census and K8 live in e4b.

## Phase 1 — correctness on the A2000 ($0, one `gpu:a2000` claim)

- **Per module, on real Mistral layers at seq 2048:** fused against unfused, for the forward output and the three
  gradients.
  - Bitwise agreement is not expected: one fused epilogue reorders the FP32 sum.
  - **Decision D3 (bar): RULED, as corrected before data.** The floor comes from neutral redraws of the UNFUSED path.
    The dense path is deterministic, so two plain unfused runs would give a floor of 0, and Phase 1 would then fail
    every time. Each redraw changes only summation order:
    - **R1:** the forward rank GEMM (x·Aᵀ) split along K = 4096 into two halves, then summed.
    - **R2:** the scale applied before the B GEMM instead of after it.
    - **R3:** the base and the adapter delta added in FP32 and then cast, instead of the delta cast and then added.
    - **R4:** the backward A- and B-gradient GEMMs split along the token dimension (two halves of the 2,048 tokens),
      then summed.
  - **Draws:** the 15 non-empty combinations of R1–R4, each one draw. The path is deterministic, so a repeat adds
    nothing. This is P129 Amendment 1's form, without its MoE-only knobs.
  - **Tensors:** for each of three layers (0, 15 and 31), on a real Alpaca batch through the frozen model, the forward
    output of `gate_proj`, `up_proj` and `down_proj`, the input gradient, and each adapter's A and B gradients.
  - **The floor, per tensor, is the WORST draw:** the largest max-|draw − unfused| and the largest relative RMS over the
    15 draws.
  - **The bar:** the fused path passes iff, for every tensor, both its max-|fused − unfused| and its relative RMS are
    at or below that tensor's floor. A tensor no draw moved has a floor of 0, and the fused path must then match it
    bit for bit.
  - The A2000 checks correctness only. No timing from it is reported or used.
- **Engagement:**
  - 96 MLP adapters are patched;
  - zero PEFT LoRA MLP forwards run in a step;
  - the qkv and o adapters are untouched (counts).
- **Streaming and checkpointing unchanged:** the prefetch events and the recompute count match the unfused path.
- **Off is bitwise:** an update with the knob off is byte-identical to today's path.
- **K8 on real Mistral:** a short fused run against an unfused one, at its initial and final steps, on both texts,
  within K8's 0.05 PPL.

The outcome is `PHASE1_PASS`, or a named failure. No paid box runs without `PHASE1_PASS` and its review.

## Phase 2 — the matched update on the RTX 5090 (paid)

- **Arms:** F0 (unfused, knob off) and F1 (fused) at the same head and recipe.
  - The order is ABBA twice: F0a F1a F1b F0b F0c F1c F1d F0d.
  - Each arm is a fresh process of 40 updates. T_arm is the median of updates 6..40.
- **Decision D4 (the 5% formula): RULED.** H = (T_F0 − T_F1) / T_F0 for each of the four block pairs, and the bar
  reads the minimum H over those four pairs (P143 Amendment 3's form). Every comparison uses the unrounded values.
  - **The four pairs:** F1a/F0a and F1b/F0b (block 1), and F1c/F0c and F1d/F0d (block 2). For each, H uses that pair's
    F0 and F1 medians.
- **NOISY:** any within-block self-pair outside [0.985, 1.015]. NOISY is a VOID and a strike; it never reads.
  - **The four self-pairs:** F0b/F0a and F1b/F1a (block 1), and F0d/F0c and F1d/F1c (block 2).
- **QUALITY:** K8 unchanged. F1 against F0 on both texts, at the initial and final steps, within 0.05 PPL. Otherwise
  the verdict is `QUALITY_FAIL`.
- **GAIN** iff H ≥ 0.05 at every pair and quality passes. Otherwise **NO_GAIN**, with each failed bar named.
  - A GAIN licenses a separate default review on a second host; never a default here.
  - **Decision D6: RULED.** A re-census of F1, if the box has time, is reported only. It is not part of the rule.

## The ceiling (a bound, not a target)

From census-9, the MLP-adapter class's exclusive device time is about 0.090·T (65.2 ms of 724.5 ms). So H cannot exceed
about 9%, even if the fusion removed every MLP-adapter cost other than the rank GEMMs. The rank GEMMs it keeps lower that
further. This bound holds on the RTX 5090 class that census-9 ran on; another card would need its own census.

## Prediction (registered; graded, never ruling)

- **What the fusion can remove:** for gate/up, one of two FP32 input casts and one rank-GEMM launch; for every
  adapter, the separate scale, add and cast-back kernels.
- **What it cannot:** the overlap.
- **Context:** DQ1 measured PEFT's unfused delta at about 9–10% of base-linear time.
- **Expected H:** 0.02–0.05, at or below the bar. NO_GAIN is the more likely outcome. A NO_GAIN closes this lever
  with its number; either way the read is cheap and clean.

## Budget — Decision D5: RULED

- **Phase 1:** $0.
- **Phase 2:** one RTX 5090, 8 arms of about 5 minutes each (including load), so about 45–60 minutes.
  - Guard 1.25 h; worst about $1.40 at $0.85/h plus the download.
  - Under the standing rule, with a reviewed kit and its own approval.

## Sequence

1. This registration merges after one review by a seat that does not consume it, then the maintainer's final pass.
2. The engine is written and reviewed, with its CPU tests.
3. Phase 1 runs on the A2000 ($0) and must give `PHASE1_PASS`, which is reviewed.
4. Phase 2's kit is reviewed, gets its own approval, and runs under the standing rule.
