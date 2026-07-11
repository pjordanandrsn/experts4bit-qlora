# Routed-subset training divergence — dose-response verdict (2026-07-11)

**Question:** when routed-subset expert-offload training diverges from whole-layer, is it
**H_FILL** (damage scales with the un-routed fill mass = 1−rf) or **H_DISCRETE** (a fixed
per-step corruption independent of how much gets filled)?

**Answer: H_FILL. Confirmed, H_DISCRETE refuted.** The divergence scales with fill mass.

## The two dose points (same model, same fix, same 150-step horizon)

| leg | seq | measured rf | fill (1−rf) | routed gap to whole-mean | floor | ratio |
|---|---|---|---|---|---|---|
| hi (Qwen3-30B) | 2048 | 0.97 | ~3% | **0.062** | 0.002 | 31× |
| lo (Qwen3-30B) | 256 | **0.688** | ~31% | **0.186** | 0.001 | 186× |

Fill went **×10.4**, the gap went **×3.0**. A fixed-corruption mechanism (H_DISCRETE) predicted
the gap would stay ≈0.06 at both points; it tripled. Provenance: `qwen30b-lo-decisive/` (this
leg, PRO 6000, prereg `3c34347`) + `conv-ab-qwen30b/` (the hi leg / primary FAIL).

## Why this is a clean result, not an artifact

- **rf transferred cleanly:** measured 0.688 vs the *registered* Phase-0 prediction 0.70
  (`prereg_qwen3_lo_decisive_design.json`, committed before the arm ran). The positive control
  holds — we genuinely reached the ~30% fill regime.
- **Floor is warm and tight (0.001):** three whole arms at 1.263/1.262/1.262 after a throwaway
  warm-up, so the 186× is measured against real noise, not a cold-arm artifact (the lesson from
  OLMoE-hi's contaminated 0.022 floor).
- **Horizon is not the driver:** within the routed run the gap is 0.224/0.189/0.186 at steps
  50/100/150 — established by step 50 and stable-to-declining, the opposite of per-step
  accumulation. Combined with the archaeology (seq64 = 16% fill × 15 steps survived at 0.018),
  the active ingredient is fill mass, not training length.
- **Cross-architecture corroboration:** the OLMoE-lo bracket showed gap 0.0615 at rf 0.97 (~3%
  fill) — matching the Qwen3-*hi* gap 0.062 at the *same* ~3% fill on a different architecture,
  host, and seq. Same fill → same gap; more fill → more gap. (OLMoE could not itself reach low
  rf — E=64 saturates — which is exactly why the Qwen3-30B leg was necessary.)

## Consequence

Routed-subset's **entire purpose** is the low-rf regime — decode and short sequences, where the
routed union is far below the full stack and the staged-bytes saving is largest. That is
**precisely where the fill damage is worst**. So the training path is not merely "opt-in with a
caveat"; the fix is **load-bearing** for routed to be usable in its target regime.

**The fix (pre-registered, unchanged):** mask the gradient-checkpointed backward's recompute
dequant to the routed rows — make the backward genuinely sparse so the un-routed rows are never
touched, eliminating the deterministic-fill corruption at its source (rather than filling
un-routed rows with a real expert's bytes, which is what leaks fill-proportional error into the
backward). Forward/decode remains bit-identical and unaffected. Until then: `whole_layer` is the
only training-validated staging; `routed` stays behind `AXOLOTL_EXPERT_OFFLOAD_ROUTED=1` with the
gate wording already noting the failed A/B.

**Scope honesty:** two dose points define a direction, not a curve shape — the `gap ~ sqrt(fill)`
fit is descriptive of exactly two measurements. A third point (e.g. seq 512 / seq 1024, rf ~0.73/0.78)
would test the shape, but the *sign and magnitude* of the dose-response — the thing that decides
whether the fix is needed — is settled.

---

## Third point (seq 512, 2026-07-11): shape is CONVEX — both frozen models rejected

The seq 512 leg ran under frozen verdict rules (`prereg_shape_verdict.json` @ `e0d1244`,
committed at 12:12:38Z with the routed arm still running; OTS-stamped). Measured: **rf 0.806 →
fill 0.194** (transfer test #3: predicted 0.730, +0.076 in the packing direction), floor
0.0010 (fourth consecutive clean n=3 warm floor — instrument validated, per A5), routed gap
**0.0843 = 84× floor** (H_FILL direction reconfirmed).

**Frozen-rule outcome:** at fill 0.194 the frozen fits predicted power 0.148 / affine 0.135;
the measured 0.0843 missed **both by ≥50× the floor**. The A1 rule mechanically fired "affine"
(nearer miss, margin 0.0137 ≥ 3×floor) — recorded as such — but per the pre-registered A2
framing (affine is physically non-viable: its intercept violates the bit-identity boundary
~50×), the meaning is: **the power law failed its third-point consistency test, and the
"affine" branch is a vacuous nearer-miss, not a fit.**

**What the three points actually show** (gap vs fill, with a whole-mean-normalized scale check
that preserves the conclusion): (0.03, 0.062) → (0.194, 0.084) → (0.312, 0.186). Monotone —
but **convex**: the upper segment is ~6× steeper than the lower. Marginal fill gets *more*
damaging, not less. This disfavors the quadrature/independent-noise mechanism (which predicts
concave √fill) and points at compounding or coherent interaction growing with fill. The A4
registered prediction for the mask-fix acceptance run (incoherent noise attenuates under
gradient averaging; coherent bias does not) stands and now discriminates more sharply.

**Unchanged consequences:** direction and magnitude are settled at three points (31× / 84× /
186× floor); routed's target regime is still where damage is worst — convexity makes that
*stronger* (the curve accelerates exactly where routed wants to operate); the mask fix remains
load-bearing. Shape claims are descriptive of fill 0.03–0.31 on Qwen3-30B at 150 steps.
Evidence: [`qwen30b-seq512-shape/`](qwen30b-seq512-shape/).
