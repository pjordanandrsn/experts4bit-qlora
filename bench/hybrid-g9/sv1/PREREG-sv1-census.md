# PREREG — SV1: current-stack decode census + dot-pad×F2 composition

Registered 2026-08-25, before measurement. Target claim under test
(Jordan; raised from 200 to **275** before any SV1 measurement):
>275 tok/s single-stream on the reference class. Arithmetic frame:
275 tok/s = 3.64 ms/step; certified points today are 7.16 ms
(0.15.1/0.21.0 defaults, from RESULTS-f2-tail's ratio on the anchor)
and ~6.59 ms with `GNF4_GEMV_DOTPAD=1` (K6-B, measured PRE-F2 — the
composition has NO receipt and is not claimed).

## Deliverables (instruments + one composition cert; no new kernels)

1. **Composition cert**: dot-pad knob ON atop current defaults
   (fused combine + fused QKV + fused append). Frame = the K6-B
   verdict verbatim (`k6b_verdict.py`, ratio bars PASS ≤0.85 /
   PARTIAL ≤0.95 against the FRESH off_a/off_b baseline, exact-length
   token identity ≥32-step divergence gate, degeneracy law). The
   baseline is the current-defaults step, NOT 7.35 — F2 moved the
   anchor class and the cert must not inherit a stale baseline
   ([[harness-defaults-are-values]] applies to anchors too).
2. **Fresh decode census** on current defaults: the F1 pipeline
   verbatim (`--torch-profile-out` + `step_budget.py` +
   `--ew-attr-out`), because the standing census predates F2 and its
   budget is stale by two treatments. Deliverable = the budget JSON
   naming the remaining slices (attn-proj launches, router, memcpy,
   elementwise, MoE GEMV) with shares.
3. **A written >275 verdict frame**: from the fresh budget, (a) the
   serving BUSY FLOOR — Self-CUDA time per graph replay, the number
   that decides whether 275 is gap/launch elimination (the training
   census's story again) or genuine compute-cutting — and (b) the
   sum of addressable slices vs the ~2.9–3.0 ms gap from the knob
   point. Pre-committed reading: if the busy floor is > 3.64 ms, 275
   is REFUTED-AS-COMPOSED for this stack and the frame must say so;
   if below, SV2+ preregs claim the gap slices the census names.
   No speed bars beyond the composition cert's — the census cannot
   fail a bar, only REFUSE on its own gates (profiler coverage,
   A/A).

## Arms (piggyback on the TR2 box after its arms, same health gate)

off_a, off_b (current defaults) → dotpad_on → census run (profiled,
separate from timing arms per the TR1 amendment). Receipts under
`bench/hybrid-g9/sv1/receipts-sv1/`.

## Refusals

Box outside the serving anchor health gate (7.39±3% probe at rent
time — the F2-era gate; the FRESH baseline may legitimately sit
below it and is judged only by A/A), A/A wider than half the
composition PASS margin, token divergence per the K6-B frame,
profiler coverage refusal from `step_budget.py`.
