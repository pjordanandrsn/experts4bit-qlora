# RESULTS — TR1: training-step census

Adjudicated 2026-08-25 by `tr1_compose.py` (amended gates, self-tests
green) on `receipts-tr1/`. One box (RTX 5090 + EPYC 9654, instance
48689229; serving graph-anchor probe 7.35 ms — dead center of the
health gate). Subject as registered: `experts4bit_qlora.train` on
Qwen/Qwen3-30B-A3B, SEQ=192, GRAD_ACCUM=4, TOKEN_BUDGET pinned 1024
(attempt 1's OOM backoff motivated the pin — see amendments), 20
timed steps × 2 runs, profiler in its own 10-step run.

## CENSUS: OK

```
anchor 51.68 s/step (~59 real tok/s)   peak GPU 24.5 GB, offload OFF
data 0.0%  forward 33.6%  backward 66.3%  loss_sync 0.0%  optim 0.1%
closure gap 0.00% both runs; half-median drift 0.66%/0.59%
A/A: shares identical to 0.1 pt; mean per-step wall delta 0.86%;
     loss trajectories digit-identical (2.269 -> 1.209 both runs)
```

The 30B **fits the 32 GB card without expert offload** (24.5 GB peak).

## The headline the census actually produced

The phase split says fwd+bwd is 99.9% of the step. The kernel window
(2 profiled steps, CUDA activity, full 2/2 window) says what is
INSIDE that:

- **Self-CUDA busy: 11.30 s over 2 steps = 5.65 s/step device work —
  the GPU is ~8–11% busy** (8.0% against the profiled-step walls of
  70–71 s; 10.9% against the unprofiled 51.68 s anchor; kernel times
  are device-side and workload-identical, the spread is the wall
  basis).
- **≥2.92M kernel launches per step** in the top-80 rows alone:
  253,578 bitsandbytes `kDequantizeBlockwise` calls/2-steps at
  3.5 µs, >500k vectorized-elementwise at ~1.5 µs, ~1.02M
  Device→Pinned micro-copies at 0.37 µs, cutlass GEMMs averaging
  2–23 µs, cub radix-sort/select router ops ×42k.
- **The account closes** (eliminate-versus-account): 2.92M launches ×
  a 15–20 µs host launch/dispatch cost ≈ 44–58 s ≈ the 52 s wall.
  The training step is **launch/host-bound**, not compute-bound. Self
  CPU time in the window: 32.1 s/2-steps — the host never catches up.

## Registered predictions, adjudicated

- **Fused QKV "<1% of step": CONFIRMED, with a measured bound.**
  Per-op attribution is unavailable (shapes dropped for profiler
  capacity — amendment 2), but ALL cutlass GEMM busy time sums to
  ~27% of Self-CUDA = ~1.5 s/step = **~2.9% of the wall**; QKV
  projections are a fraction of that. Not a registrable treatment.
- **Expert fwd+dgrad "5–25% config-tuning lever": REFUTED AS
  FRAMED.** Expert device work is small (dequant 7.9% of busy =
  ~0.9% of wall); tuning launch configs of kernels that are 1.5–4 µs
  each moves nothing. The census surfaced the real lever instead:
  **~89–92% of the wall is host/launch overhead from the per-expert
  micro-kernel chain** (bnb dequant → tiny GEMM → elementwise →
  DtoH, per expert, per layer, per micro-batch).

## The TR2 lever (named, not registered here)

Collapse the launch storm: route training expert fwd/dgrad through
grouped kernels (the gnf4 grouped machinery this program already
certified for serving — one launch over all experts' rows instead of
per-expert chains), and/or CUDA-graph the step. The ceiling if device
work were perfectly packed is ~5.65 s/step (~9x). TR2's prereg will
register the treatment, bars derived from this census's anchor and
busy floor. Predictions this file commits to before TR2 is written:
the win is bounded above by 9.1x, and any treatment that does not cut
LAUNCH COUNT by ≥10x will not reach 2x.

## Failure record (all receipts committed)

- Attempt 1: 8-step shape-recorded profiler inside census run A →
  container OOM-kill at the 170 GiB cgroup cap (peak == limit); also
  its overhead perturbed run A (+55%/step). → profiler quarantined to
  its own run (amendment).
- Attempt 2 p2: names-only CPU+CUDA at active=4 OOM-killed during
  `key_averages()` AFTER training → CUDA-only, active=2 (amendment
  2); p3 completed with the full 2/2 window.
- Original steadiness statistic (max−min < 10%) refused run A at 15%
  while the per-step pattern reproduced across runs at 0.86% —
  re-derived as the half-median trend test + a stricter per-step A/A
  gate (disclosed amendment; the direction admits these receipts and
  the self-test proves genuine drift still refuses).

## Scope

One box, one model family, one batch recipe (TOKEN_BUDGET=1024,
SEQ=192, GRAD_ACCUM=4, gradient checkpointing ON — backward includes
recompute, hence 2:1 bwd:fwd). The launch-storm finding is structural
(per-expert kernel chains), not box-tuning, but the 8–11% busy figure
is this recipe's number; larger token budgets pack better and TR2's
baseline arm re-measures at its own registered config.
