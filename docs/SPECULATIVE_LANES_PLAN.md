# SPECULATIVE LANES PLAN — POST-S9 REVISION (2026-07-05)

```
status:    exploratory program — SECOND-CLASS to the confirmatory queue by construction.
           Consumes byproducts of scheduled runs; never blocks T1–T8 or the ladder
           re-pin; never mints precision-quality claims denominated in G while S9 holds.
cites:     PREREG_AMENDMENT_LADDER_REPIN_n1024.md (telemetry + branch table)
           AGENT_HANDOFF_POST_AUDIT_20260705.md (queue, quarantine)
           D1/D2 n=64 report (SD-clustering, fp8 null, bitwise D2)
owner:     Jordan        executor: coding agent
two-stage rule: the n=1024 set is CONFIRMATION for the one preregistered primary and
           EXPLORATION for everything below. Any lane that graduates gets its own
           preregistered follow-up on a fresh set before a confirmatory claim ships.
```

## 0. The Spine (build once; all lanes are analyses on it)

One joined dataset per model:

```
per example i (n = 1024, pinned):
  loss_i(mode)                 — scheduled (amendment, 12–13 passes)
  routed_sets_i(mode, layer)   — scheduled (amendment REQUIRED telemetry)
  margins_i(layer)             — scheduled (amendment PREFERRED flag → set the flag)
per format p:
  W_RMS(p) = relative RMS of packed-dequantized weights vs bf16 reference
                               — NEW, zero-GPU (one pass over tensors, CPU/GPU-trivial)
per adapter a (already on disk from portability runs):
  loss_i(a, serve-mode), routed_sets_i(a, serve-mode)
                               — NEW, 2–3 eval passes with telemetry
```

New GPU beyond scheduled: ~2–3 passes (S-B) + optional gated items. Everything else
is joins and statistics on vectors already committed to disk.

Contamination note: exploratory analyses on the confirmation set are permitted and
labeled; the primary contrast is untouched by them (preregistered before any of this).

---

## S-A — The response curve (PROMOTED: first bet)

**Claim under test.** Per-format functional divergence from the bf16 reference is a
single response curve: x = measured perturbation magnitude, y₁ = routing flip rate
(1 − Jaccard vs bf16), y₂ = per-example loss disagreement (paired sd vs bf16). If the
curve is coherent, G, the co-adaptation penalty, and the chaos exponent are readouts
of one margin distribution at different perturbation scales.

**New-data support.** The n=64 SD-clustering IS the predicted signature: {int8, bf16,
fp16} tight (0.013–0.019), {nf4, fp4, fp8} scattered (0.05–0.09) — including from
each other, which is the different-subsets-flipped fingerprint.

**Method.** Compute W_RMS(p) statically. Plot (W_RMS, flip-rate, paired-sd) for
{int8, fp8, nf4, fp4} vs bf16; fp16 enters as the x≈0 point whose nonzero sd (0.0188
at n=64) calibrates the activation-compute channel — the curve's intercept. D3's
branches later add the ~1e-6 (kernel drift) and order-1 (dropout) points.

**Committed predictions.**
- W_RMS ordering: int8 (~3e-3) < fp8 (~2e-2) ≲ nf4 ≈ fp4 (~3–5e-2) — i.e., fp8's
  weight perturbation lands nearer the 4-bit formats than int8. Bits are not the axis.
- Rank correlation between W_RMS and paired-sd-vs-bf16 across the 5 formats: perfect.
- Jaccard(int8, bf16) ≥ 0.97; Jaccard(nf4, bf16) ∈ [0.75, 0.92];
  Jaccard(fp8, bf16) within ±0.05 of Jaccard(nf4, bf16).
- Spearman corr(|d_i|, per-example flip count) > 0.3 for cross-cluster pairs;
  ≈ 0 within the tight trio.

**Graduation.** All four predictions hold → preregister the cross-architecture test
(Qwen3 sentinel ladder, 6 passes + telemetry on the A100) asking whether the curve
transfers under the model's routing config. **Kill.** W_RMS fails to predict the sd
clusters, or flip rate is uncorrelated with |d_i| → the clusters are kernel-path
artifacts, not routing physics; report and stop.

---

## S-B — Adapters steer routing (CHEAPENED: measure-first)

**Claim under test.** Part of what a LoRA adapter learns on a frozen-router MoE is
routing edits — moving tokens across expert boundaries — not only expert-output edits.

**New-data support.** Adapters, pinned set, and telemetry all exist; marginal cost
collapsed to 2–3 eval passes. VSRAQ-line prior: perturbing inputs to routers moves
routing; adapters perturb the residual stream upstream of every router by design.

**Method.** Eval nf4-trained and int8-trained adapters on their train-precision base
with routing logged; compare routed sets base-vs-adapted per example.

**Committed predictions.** J(adapted, base) ∈ [0.85, 0.97]; adapted-routing shift
LARGER than int8-vs-bf16 shift, SMALLER than nf4-vs-bf16 shift; per-example routing
movement correlates positively with per-example adaptation gain (|Δloss_i|).

**Graduation.** Correlation holds → the mismatch mechanism gains a second term
(adapter moves routing computed on base A; base B moves it differently), and the
routing-pinned-serve experiment is promoted with a mechanism-complete story.
**Kill.** J ≈ 1.0 everywhere → adaptation is expert-content only here; one clean
descriptive result, lane closed. (Either way the measurement ships — it is the first
base-vs-adapted routing-shift number for this stack.)

---

## S-C — Fragility gradient → per-expert bit-width

**Claim under test.** Per-(layer, expert) fragility — margin contribution + flip
attribution — predicts per-expert bit tolerance, collapsing mixed-precision search
to a computation.

**Method (spine joins only).** For the top-decile |d_i| examples, attribute which
(layer, expert) pairs flipped or were routed; build the fragility map from margins
(flag telemetry) + flip attribution. No new runs.

**Committed prediction.** Flipped pairs in the top-|d_i| decile concentrate ≥ 2×
over uniform incidence.

**Graduation.** Concentration holds → preregister one mixed-precision cell: hold the
top-fragility experts at int8, rest at nf4; predict it recovers ≥ 50% of whatever
G the re-pin certifies, at ≤ 15% of int8's memory premium. **Gate:** the follow-up
runs only if the re-pin's primary clears (a mixed-precision dial needs a certified
gap to recover). **Kill.** Fragility is diffuse (echoing the train-phase stall
profile) → per-expert precision is not the dial; report, close.

---

## S-D — fp8 / routing-perturbation regularizer (RE-GATED)

**Status change.** The original trigger FAILED at n=64 (fp8 − bf16: |t| = 0.96).
Lane is dormant behind the amendment's preregistered branch: **fp8 − {nf4, fp4}
replicates ≥ 3σ at n=1024.**

**If gated in.** Two short OLMoE training runs (~30 min each): bf16 base, Gaussian
noise injected at router logits (σ from the measured margin distribution — set to
target the same flip rate the spine measures for fp8), vs no-noise twin, paired seed.
**Prediction (conditional).** Router-noise run matches or beats the twin on held-out.
**Kill.** fp8 branch does not fire at n=1024 → lane closes without the probe;
the n=64 "best mean" is recorded as sampling scatter.

---

## S-E — Chaotic early, contractive late (GATED ON D3)

**Claim under test.** Twin divergence during fine-tuning grows then saturates; a
measurable transition step exists past which the basin is decided.

**Method.** Extend whichever D3 branch supplies a certified perturbation source into
one K-step and one full-run (100-step) twin pair, logging per-step adapter-weight
divergence and routed-set flip counts (the flip-annotated divergence curve — the
object the prior-art search confirmed nobody has). Prior: the 2023 replication found
duration dependence.

**Committed prediction (weak, honest).** Divergence saturates within the 100-step
run: ~40%. Flip events dominate divergence growth once the first fork fires: ~70%.

**Gate.** Runs only after D3 lands and only as ONE twin pair (~30 min).
**Graduation.** Saturation observed → the transition step becomes a reproducibility
recommendation ("train past step S before comparing methods") and a paper-shaped
result. **Kill.** Monotone growth to end-of-run → the exponent stands alone as
calibration (still pays into the flat-axis headline); lane closes.

---

## S-F / S-G — Parked (fingerprints; scaling axis)

S-F (routing signatures as provenance/security primitive): no dedicated work; S-B's
base-vs-adapted Jaccard is the feasibility datum, recorded in passing. Owed before
any claim: the MoE attack-surface prior-art search. S-G (fragility as scaling axis):
requires a model sweep this program does not have; the optional Qwen3 sentinel from
S-A graduation yields n = 2 — a ratio, not a law. Both stay parked.

---

## Pre-registered surprise bet (ledger entry, committed now)

Revised order, to be graded against outcomes: **S-A > S-C > S-B > S-E > S-D**.
This REVERSES the pre-data ordering (which led with fragility→bit-width and
duration→saturation); the reversal is driven by the SD-clustering evidence and the
collapse of S-A's cost to near zero. Prior forecast rule applies: no significance
predictions here without n in hand — all prediction thresholds above are stated
against the n=1024 SEs from the amendment's power table.

## Budget

| item | cost | status |
|---|---|---|
| W_RMS per format | CPU minutes | new, unconditional |
| Spine joins + all S-A/S-C analyses | CPU minutes | on scheduled data |
| S-B adapted evals | 2–3 × ~15 min A5000 | new, unconditional |
| Qwen3 sentinel ladder | 6 passes A100 | gated on S-A graduation |
| S-D probe | 2 × ~30 min A5000 | gated on fp8 branch |
| S-E twin pairs | ~30–60 min A5000 | gated on D3 |

## What will not be claimed

- Nothing denominated in G while S9 holds; S-C's follow-up explicitly requires the
  re-pin's primary to clear.
- No confirmatory claim from any spine analysis without a fresh-set follow-up
  (two-stage rule).
- No cross-architecture curve claim from one model; the Qwen3 sentinel yields a
  transfer TEST, not a universality claim.
- No security/provenance claims (S-F) without the owed prior-art search.

---

## Executor's filing notes (added at commit time; not part of the handoff text)

- In-repo name mapping: `PREREG_AMENDMENT_LADDER_REPIN_n1024.md` =
  `docs/NULL_LADDER_1024_AMENDMENT.md`; `AGENT_HANDOFF_POST_AUDIT_20260705.md` =
  `docs/POST_AUDIT_WORK_QUEUE.md`; the D1/D2 n=64 report =
  `runs/results/postaudit/null_ladder_per_example.md` (commit 5e8f831).
- The amendment's PREFERRED router-margin flag is already ON in the scheduled
  manifest (`runs/job_manifest/null_ladder_1024_jobs.jsonl`).
- Sequencing honored: the 13 confirmatory ladder passes run first; W_RMS + the two
  S-B passes run after them on the same pod (same architecture); all spine analyses
  are controller-side joins.
- S-B adapters: the seed-0 portability adapters on the volume
  (`/workspace/matrix/olmoe_mode_adapters/{nf4,int8}/adapter_best.pt`, provenance
  sidecars alongside), served resident on their train-precision base.
- Note on "zero-GPU" W_RMS: the bnb quantize/dequantize kernels are CUDA-only, so
  W_RMS runs as one trivial GPU job (minutes); the "zero-GPU" intent (no new eval
  passes) is preserved.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T18:07:59Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `43c48fe5753b829d57917bcf2d296ddfc235e4c7e9f8a2079dff59bb5df6acf6` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[o~&o*$?O=O~@*+#!]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|       ..      .o|
|       .. . . ...|
|        .= + o o.|
|       .. + + +o.|
|        S   .++==|
|         . ..o=+B|
|            .+++B|
|             ++=X|
|           .o.+OE|
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info SPECULATIVE_LANES_PLAN.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify SPECULATIVE_LANES_PLAN.md.ots SPECULATIVE_LANES_PLAN.md` succeeds against the on-disk bytes.
- Anchor file: `SPECULATIVE_LANES_PLAN.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
