# DQ11 MLP-adapter BUILD — DRAFT registration (#1083)

**Status:** DRAFT for review. Not registered; nothing runs from it. Each **Decision** below is a proposal that the
maintainer rules on before this becomes a registration. Merging this draft costs $0 and licenses nothing.

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

## The subject — Decision D1 (scope)

**Proposed.** A training engine in e4b, `enable_dense_mlp_lora_fuse(model)`:
- **Knob:** `E4B_DENSE_MLP_LORA_FUSE`, off by default. With it off, the path is bitwise today's PEFT path.
- **Targets:** the PEFT LoRA modules on each decoder layer's MLP (`gate_proj`, `up_proj`, `down_proj`; 96 adapters on
  Mistral-7B). Its count is returned, and an unexpected module is refused rather than skipped.
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

**Alternatives for the ruling:**
- (b) epilogue-only, with no shared gate/up input;
- (c) an Unsloth-style LoRA_MLP that includes SwiGLU. It reaches past C_MLP into other classes, so it is not proposed.

## Fixture

- **Carried from the L science path, unchanged:**
  - Mistral-7B-v0.1 at `27d67f1b`;
  - seven targets, r16 / α32 / dropout 0, FP32 adapters;
  - NF4 bases with BF16 compute;
  - seq 2048, mb1/acc1, non-reentrant checkpointing, TF32 off;
  - streamed with `train_prefetch`, 40 AdamW updates, seed 3407.
- **New:** e4b at the merge head that carries the engine. The science closure records that head in place of
  `13b55990`.
- **Decision D2 (code location):**
  - The engine lives in e4b (`experts4bit_qlora/engines/dense_lora_fuse.py`) and patches only the PEFT `LoraLayer`
    forward of the MLP modules.
  - The alternative is a Loggetta-side patch, which would move a second pin.

## Phase 1 — correctness on the A2000 ($0, one `gpu:a2000` claim)

- **Per module, on real Mistral layers at seq 2048:** fused against unfused, for the forward output and the three
  gradients.
  - Bitwise agreement is not expected: one fused epilogue reorders the FP32 sum.
  - **Decision D3 (bar):** the unfused path's own neutral floor, per tensor (two unfused runs with a neutral redraw).
    This is P129 Amendment 1's lesson: a fixed op-level tolerance failed on a whole step.
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
- **Decision D4 (the 5% formula):** H = (T_F0 − T_F1) / T_F0 for each of the four block pairs, and the bar reads the
  minimum H over those four pairs (P143 Amendment 3's form). DQ11's own form ("both repetitions") is the alternative.
- **NOISY:** any within-block self-pair outside [0.985, 1.015]. NOISY is a VOID and a strike.
- **QUALITY:** K8 unchanged. F1 against F0 on both texts, at the initial and final steps, within 0.05 PPL. Otherwise
  the verdict is `QUALITY_FAIL`.
- **GAIN** iff H ≥ 0.05 at every pair and quality passes. Otherwise **NO_GAIN**, with each failed bar named.
  - A GAIN licenses a separate default review on a second host; never a default here.
  - **Decision D6:** whether a GAIN must also show, in a re-census, that C_MLP fell. Proposed: reported only.

## Prediction (graded, never ruling)

- **What the fusion can remove:** for gate/up, one of two FP32 input casts and one rank-GEMM launch; for every
  adapter, the separate scale, add and cast-back kernels.
- **What it cannot:** the overlap.
- **Context:** DQ1 measured PEFT's unfused delta at about 9–10% of base-linear time.
- **Expected H:** 0.02–0.05, at or below the bar. NO_GAIN is the more likely outcome, and a NO_GAIN closes the MLP
  BUILD with its number.

## Budget — Decision D5

- **Phase 1:** $0.
- **Phase 2:** one RTX 5090, 8 arms of about 5 minutes each (including load), so about 45–60 minutes.
  - Guard 1.25 h; worst about $1.40 at $0.85/h plus the download.
  - Under the standing rule, with a reviewed kit and its own approval.

## Open for the maintainer's ruling before registration

- D1: scope;
- D2: code location;
- D3: Phase 1's bar;
- D4: the 5% formula and its repetitions;
- D5: budget and box;
- D6: whether the re-census must confirm.

No code is written until D1 and D2 are ruled.
