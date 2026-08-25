# PREREG — TR1: training-step census (the training-port gate)

Registered 2026-08-25, before any measurement. Directive: port the
serving campaign's wins to the training side. Discipline: **census
before treatments** — no training treatment gets registered until this
census says what a step is actually made of. This prereg registers
instruments and refusal conditions only; it carries NO speed bars and
ships NO behavior change.

## Why a census first

The serving campaign's two portable mechanisms have opposite prior
plausibility at training shapes, and both priors are guesses:

- **Fused QKV** won 0.120 ms at M=1 because decode is launch-bound.
  Training M is thousands of rows; the same GEMMs are compute-bound
  and the predicted win is <1% of step. Prediction registered here so
  the census can confirm or embarrass it.
- **Grouped expert GEMM/dgrad configs** (the K1 analogue) target the
  bulk of training FLOPs — `_HybridProjFn` forward + backward through
  the 4-bit stacks. Whether tuning is available depends on whether
  today's configs are M=1-shaped defaults misapplied at large M, or
  the original large-M tuning still fits. The census decides; the
  eliminate-versus-account law forbids summing guesses.

## Subject

One QLoRA fine-tune step of `python -m experts4bit_qlora.train` on
the campaign's reference class (RTX 5090 rental, serving graph-anchor
7.39 ± 3% as the box-health gate — no training anchor exists yet;
**establishing one is a TR1 deliverable**).

- Primary config: `MODEL=Qwen/Qwen3-30B-A3B` (campaign continuity),
  `TOKEN_BUDGET=2048`, `SEQ`/`R`/`TRAIN_EXPERTS`/`TRAIN_ATTENTION` at
  trainer defaults, `OFFLOAD_EXPERTS=1` if VRAM requires (recorded
  either way — offload traffic is part of the budget, not a confound
  to hide).
- Fallback if the primary cannot run on the class box: the trainer's
  reference `allenai/OLMoE-1B-7B-0924`, recorded as a scope narrowing
  (findings then port to Qwen3 only after a spot-check).

## Instruments

1. **Phase brackets** (wall-clock, CUDA-event-fenced): data/tokenize,
   forward, backward, optimizer step, offload H2D/D2H if enabled.
   Sum-of-phases vs step wall must close within 5% or the census
   REFUSES (unaccounted time is a finding, not a rounding error).
2. **Kernel budget**: `torch.profiler` over a 4-step window in its
   OWN short run, parsed by the existing
   `bench/hybrid-g9/f1/step_budget.py` coverage-gated pipeline
   (reused, not rewritten). Amended before any composed receipt
   existed: the first arms put an 8-step shape-recorded window INSIDE
   census run A -- it drove the container into its 170 GiB cgroup cap
   (SIGKILL at step ~10, cgroup peak == limit), and its overhead was
   perturbing the very phases the A/A compares. The profiler run is
   excluded from the A/A pair by design.
3. **Steady-state gate**: first N warmup steps dropped; the timed
   window's step-time spread must be < 10% of median or REFUSE
   (compile/caching still in flight).
4. A/A: the full census runs twice; phase shares must agree within
   3 points absolute per phase or REFUSE.

## Deliverables (no speed bars in TR1)

- `tr1_budget.json`: phase split + top-20 kernels with shares.
- The training anchor: median step ms for the registered config on
  the class box (becomes the TR2 baseline).
- A written verdict on each port candidate's addressable share:
  fused-QKV's projection-GEMM share, and the expert forward+dgrad
  share with its current launch configs identified.
- TR2 prereg (treatments + bars) only after this merges with
  receipts.

## Refusals

Missing phase closure, non-steady window, A/A phase drift, box
outside the health gate, or a training run that diverges/NaNs inside
the timed window (loss must be finite every step; a broken run's
budget is not a budget).
