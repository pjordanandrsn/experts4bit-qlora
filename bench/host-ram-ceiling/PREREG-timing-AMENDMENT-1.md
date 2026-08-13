# Amendment 1 to PREREG-timing — the gate was mis-specified

**Written 2026-08-13, after the OLMoE round set and BEFORE any re-run.** Stamped
pre-data for the runs it governs, so the corrected gate cannot be tuned to a result it
has already seen. The original gate and the run it failed are left intact in
[`PREREG-timing.md`](PREREG-timing.md) and reported as failed.

## What the original gate said, and why it is wrong

> *"if `host_self` spread exceeds ±8%, report 'indistinguishable at this resolution' and
> no ratio"*

That uses the control's **spread** as a proxy for whether the harness is precise enough,
and then applies it as an absolute threshold regardless of how large the effect is. It
produces a nonsense verdict on the data it was written for:

| | effect | control | original gate |
|---|---|---|---|
| load | **3.39×** [3.05–3.75] | 0.995, spread 8.4% | "not resolvable" |
| step | **1.331×** [1.309–1.423] | 0.982, spread 19.9% | "not resolvable" |

Calling a **239%** effect unresolvable because a control moved 8.4% is not a conservative
reading; it is an incorrect one. The gate confuses *resolution* with *significance*.

## The corrected gate

A ratio is reportable when **the effect's per-round range does not overlap the control's
per-round range**, both computed as within-round paired ratios over the same scored
rounds. The control's range *is* the resolution; an effect that clears it is resolved, and
an effect inside it is not — independent of how wide the control happens to be.

Reported alongside every ratio, always:
- the control's median and range, so a reader can apply their own threshold;
- the number of scored rounds and scored steps;
- explicitly, whether the ranges overlap.

**This is a stricter test than "control within ±8%" in the case that matters** — a wide
control now *widens* the band an effect must clear, rather than voiding the run.

## Protocol change for the re-run

The original protocol's control was noisy because each run scored only **6 steps** after
2 warmup, and step-to-step variance dominates at that count (observed range within one
run: 1.44–3.85 s). The re-run uses **12 scored steps after 4 warmup**, and **5 scored
rounds** instead of 3.

This is a change to precision only. **No threshold moves with the data**, the arms are
unchanged, and the corrected gate above is fixed now, before the re-run produces a single
number.

## What the failed run still establishes

Nothing is discarded. The original round set stands as a completed run under its own
stated rule, and its verdict — *not resolvable under the registered gate* — is reported as
such. If the re-run reproduces the same effects with a control that clears the corrected
gate, the pair of runs is stronger evidence than either alone. If it does not reproduce
them, the original effects were noise and the failed gate was right for the wrong reason.

**Pre-committed prediction for the re-run:** load host/arena **3.0–3.8×**, step arena/host
**1.25–1.45×**, control range narrowing to under 10% on step. If the step effect's range
overlaps the control's under the corrected gate, the honest output is still
"indistinguishable", and no step-time number ships.
