# RESULTS — TR2: grouped expert kernels for the training step

<!-- BASELINE RESTATEMENT (2026-08-27). The 13.47x below is CORRECT as
measured and is NOT withdrawn: the grouped arm reproduces at 3.7 s/step on a
fresh box. But its BASELINE has moved. transformers v5 ships
grouped_mm_experts_forward and fused the per-expert loop upstream, so the same
bnb arm that read 50.86 s/step here measures 26.6 s/step on transformers 5.5.0.
Against a CURRENT baseline the same-box speedup is 7.19x. Anyone who reruns
13.47x on a current stack will get about half of it. See "Baseline restatement"
at the end of this file. -->

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


## Baseline restatement (2026-08-27): 13.47x is 7.19x on a current stack

Appended, not edited: the original verdict is left intact because it is
correct for the stack it was measured on.

**The treatment did not regress.** Re-run on a fresh RTX 5090 (vast 48803545,
driver 595.84) with transformers 5.5.0, bnb 0.50.1, gnf4 0.16.0, the TR1
recipe verbatim:

| arm | s/step | tok/s | peak VRAM | held-out eval |
|---|---|---|---|---|
| base (bnb) | **26.6** | 116 | 24.36 GB | 1.0093 |
| grouped (`TRAIN_ARENA`) | **3.7** | 846 | 24.37 GB | 1.0118 |

The grouped arm reproduces this document's 3.77 s/step at 3.7. **The baseline
halved**: 50.86 -> 26.6 s/step, because transformers v5 ships
`grouped_mm_experts_forward` and fused the per-expert expert loop upstream.
Roughly half of the published multiple is now upstream's work, not ours.

Recipe validity is confirmed by eval parity with this document's own figures
(base 1.010 / grouped 1.009; re-measured 1.0093 / 1.0118), and the token
denominator is now measured rather than inferred: the trainer's own log prints
`116 tok/s` at `26.6 s/step` = **3,086 tokens/step**.

**How to state it:** *7.2x over a current-transformers bnb baseline, same-box,
with held-out eval parity.* That survives a rerun. `13.47x` does not.

### Two caveats that bound even the 7.19x

1. **Same-box only.** Identical config measured 3.7 s/step on one RTX 5090 and
   6.1 s/step on another — **1.65x**. Use `bench/train-anchor/` to class a box
   before comparing anything; "RTX 5090" is two hardware classes.
2. **Not a competitive claim on its own.** Current Unsloth (2026.8.21 +
   unsloth_zoo 2026.8.15, transformers 5.5.0) *can* now train 4-bit MoE QLoRA
   -- their docs saying BitsandBytes cannot are stale. Measured same-box at
   identical 321,257,472 trainable params: **grouped 3.7 s/step vs Unsloth
   4.3328**, so we lead by 1.17x, at 24.37 GB against their 21.55 GB.


## Fit addendum (2026-08-29): `TRAIN_ATTN_4BIT` closes half the VRAM gap at 1.3% cost

Appended, not edited, same discipline as above. The 21.55-vs-24.37 GB caveat
now has a shipped answer (PR #299): store the FROZEN attention projections in
bnb NF4 before the LoRA wrap — Unsloth's own `load_in_4bit` trick, applied
surgically. Measured on the audit recipe, base arm reproducing this
document's numbers exactly (receipts in
[receipts-attn4bit/](receipts-attn4bit/), 2×2 interleaved rounds):

| arm | load GB | peak GB | s/step | held-out eval |
|---|---|---|---|---|
| grouped (as above) | 20.03 | 24.37 | 3.850 | 1.0099/1.0101 |
| grouped + `TRAIN_ATTN_4BIT=1` | **18.69** | **23.03** | 3.900 (**1.013×**) | 1.0142/1.0152 |

**Fit gap to Unsloth: 2.82 → 1.48 GB, with the 1.111× speed lead kept.**

Why training tolerates what serving does not: the forward runs at
M = seq × batch (hundreds of rows per expert group), the regime where
dequant-then-matmul wins the fused/dequant crossover; serving decode is
M = 1, where the same quantisation measured 3.4 ms/step *slower*. One
curve, two regimes, both verdicts predicted before measurement
(grouped-nf4-gemm `bench/sm120-census/`).

The remaining 1.48 GB is embeddings + lm_head bf16 plus arena layout —
deliberately not taken: both sit in the loss path even when frozen, and
NF4 lm_head measured **+0.40 ppl** over an 8,192-token teacher-forced
window. Closing it would trade training quality for fit; that is a
different decision than a free win, so the flag stops here.
