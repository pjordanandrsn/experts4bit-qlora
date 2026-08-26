# RESULTS — TR2: grouped expert kernels for the training step

Adjudicated 2026-08-26 by `tr2_verdict.py` (amended A/A gate,
self-tests green) on `receipts-tr2/tr2_report.json`. Box: RTX 5090 +
EPYC 9654, instance 48709950 (serving anchor probe 7.25 ms — the
post-F2 class). Recipe as registered: the TR1 subject verbatim,
TOKEN_BUDGET pinned 1024.

## VERDICT: PASS — 13.47×

```
step 3.77 s vs 50.86 s (base sanity: inside TR1's 51.68 anchor ±10%)
launches/step 2,920,655 -> 126,325 (23.1× cut)
evals: base 2.483->1.010, hybrid 2.482->1.009 (identical learning)
loss trajectories finite throughout; adapters saved both arms
```

`TRAIN_ARENA` ships as the documented default path for arena-holding
models: training the 30B QLoRA recipe moves from ~59 to **~800 tok/s
class** on the reference box, a full fine-tune epoch of the TR1
recipe from ~17 minutes of stepping to ~75 seconds.

## The registered model check — and a falsified bound

The additive launch model survives in direction (both cuts large,
consistent) but its CEILING premise is falsified, in the treatment's
favor and outside the registered falsification cells: the hybrid wall
(3.77 s) sits BELOW the assumed 5.65 s busy floor — implied per-launch
host cost goes NEGATIVE (−14.8 µs). The floor was measured on the
bnb path and assumed path-invariant; the grouped path's DEVICE work
is itself far smaller (one grouped GEMM per projection versus 253k
dequants + tiny GEMMs + elementwise chains). RESULTS-tr1's "bounded
above by 9.1×" was therefore a wrong (conservative) bound, recorded
here as such.

## Amendment record (disclosed, both directions tested)

The same-workload A/A gate was purely relative (2% of median) and
refused the hybrid pair at 6.2% — while the receipts show per-step
jitter is ~250 ms ABSOLUTE regardless of step duration (hyb 243 ms
vs base 251 ms): the treatment shrank the denominator 13×, and the
hybrid pair's absolute agreement BEAT the baseline's own. The gate
now carries an absolute floor of 1.5× the base pair's measured delta
(same box, same recipe); the base pair itself still binds at the
relative gate, and the self-tests prove a genuine systematic delta
refuses at any floor (the first version of that self-test cell was
itself vacuous — refused via closure — caught in review and fixed).

## Failure record

- hyb_a attempt 1: OOM at the tier-build PEAK (bnb + arena stacks
  co-resident) — the #259 review's prediction hitting live after a
  post-enable release fixed only the steady state.
- hyb_a attempt 2: `Parameter.data = <meta>` rejected by autograd —
  fixed by REPLACING the storage objects with shape-preserved meta
  twins, and the mechanics are now behaviorally tested on the real
  `ExpertsNbit` class (the two broken variants had source-guard-only
  coverage).

## Scope

One box, the TR1 recipe (SEQ=192, GRAD_ACCUM=4, TOKEN_BUDGET=1024,
grad checkpointing ON), Qwen3-30B-A3B, all-VRAM arena placement.
TOKEN_BUDGET was pinned for comparability, not optimality — the
budget knob re-opens on top of this step mechanics as its own cycle.
