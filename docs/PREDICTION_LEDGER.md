# Prediction ledger

Graded predictions and adopted forecasting rules, one entry per grading event, newest first.
Entries are recorded verbatim from their source documents at filing time; gradings are never
edited after the fact — corrections are new entries.

## 2026-07-06 — N2 routed-stream Phase 0–1 (reconstruction) GRADED

Reconstruction of v3 Phase 0–1 (`docs/N2_PHASE01_RECONSTRUCTION.md`; not the real v3).
Rev1 had a margin-hook red (OlmoeTopKRouter ≠ nn.Linear → empty margins, vacuous P-B); fixed
(take router_logits from the tuple) and re-run.
- **Core: SPARE all cache lanes** (nf4 +34.3% / int8 +33.7% / bf16 +43.1% at 2 GB) — decode
  locality low (Jaccard ≈0.30) but working set cacheable. **A2's "kill nf4" prior CORRECTED.**
- **O2 margin-aware eviction: no benefit** (≈ plain LRU, bf16 worse) — clean negative.
- **P-B1 (corr(near-margin, Jaccard) < −0.40): BORDERLINE** — −0.415/−0.435 (HOLD),
  −0.350 (adapter, FAILS); all negative ~−0.4 but n=16 layers makes it non-robust.
- **P-B2 (churn quartile ≥ 1.5×): FAILS** (1.09–1.10×).
- **O1 byproduct:** base-vs-adapter decode routing Jaccard 0.771 (< eval-set 0.942) —
  adapters move routing more under decode, reinforcing S-B.
Net: build the simple working-set LRU; skip margin-aware complexity. `runs/results/postaudit/n2b/`.

## 2026-07-06 — N3 fragility attribution GRADED (mixed-precision cell gate)

Committed S-C gate (ba23461, pre-data): top-decile |d_i| flips concentrate ≥ 2× over uniform.
**Literal gate MET** (top-10% of pairs hold 52% = 5.2× uniform) — **but the fragility-specificity
control FAILS**: the least-fragile decile is equally concentrated (55%), so routing flips localize
to the same experts regardless of precision fragility. "Top-fragility experts" is not a set
distinct from "top-flip experts," and they don't track |d_i|. **N3 CLOSES** on the control
(transparently — the literal committed gate passed; it was under-specified, concentration alone
being non-diagnostic). Per-expert precision is not the dial. Fragile examples do flip more in
total (2341 vs 1591) but on the same expert distribution. `runs/results/postaudit/n3_fragility_attribution.md`.

## 2026-07-06 — N1 routing-pinned serve GRADED (next-campaign lane, first result)

Committed odds 30/45/25 (`docs/N1_ROUTING_PINNED_SERVE.md`). **Middle branch (0 < R < 0.50,
45%) HIT.** R = (L1−L2)/G = **0.365 ± 0.092** (|t| ≈ 4.0): pinning the int8-served nf4-trained
adapter's routing to the nf4-home reference recovers ~37% of the certified precision gap —
a real, significant lever (confirms S-B's serving relevance) but a minority of the forfeit;
the rest is value-space co-adaptation pinning can't fix. Pin engaged fully (16384/16384
overrides). One adapter/host/eval — positive-but-partial → licenses a seed-replicated
follow-up before any serving claim, does not itself ship one. `runs/results/postaudit/n1_routing_pinned_result.json`.

## 2026-07-06 — Reviewer closure claim CORRECTED (operator-caught)

"Queue state after this document: empty" (DIVERGENCE_ONSET_PROBE) and the reviewer's
"the queue reads empty" **overstated closure**: the filed-not-graded index below held
P-B1–P-B4 open, and two graduation clauses had fired (S-B → routing-pinned serve; S-C
precondition → G cleared). Correct statement: the **confirmatory** queue is empty; three
licensed continuations remain (N1 routing-pinned serve, N2 routed-stream T7, N3 mixed-precision
cell — `docs/NEXT_CAMPAIGN_LANES.md`). Caught by the operator, not the apparatus — the
three-verb test does not check claims of absence. **Rule adopted:** a closure claim is a claim;
grade it against the filed-not-graded index before making it.

## 2026-07-06 — Divergence-onset probe GRADED (the final gate)

Committed odds 30/55/15 (`docs/DIVERGENCE_ONSET_PROBE.md`). **Branch 2 (55%) HIT** in its
strongest form. The resident-twin null (A–B) diverges at the same step (2) and to a marginally
larger magnitude (final A–C/A–B = 0.89) than the placement pair (A–C): the same-host 150-step
bf16 0.0108 gap is run-to-run chaos (nondeterministic atomics amplified over 150 steps),
**not a placement effect**. Step 1 bitwise (D3 respected); routing flips follow weight
divergence (step 3 vs step 2), they don't cause it. Branch 1 (static, clean — ruled out) and
branch 3 (other) did not fire. Placement question closed and stays closed.
Harness: two pre-fix reds under Q4 (rev1 fp64-RAM OOM, rev2 network-memmap page-cache OOM);
rev3 fixed-subset sketch produced the result. `runs/results/postaudit/divergence_bf16_result.json`.

## 2026-07-05 — GRADED at drain (n=1024 + rev3 + spine extras)

**Amendment §4 branch table (confirmatory, `runs/results/postaudit/null_ladder_1024.md`):**
- PRIMARY nf4−int8 **CLEARS**: G_int8 = +0.01657 ± 0.00227, |t| = 7.29; Wilcoxon z = +12.43.
  S9 branch 1 — the precision gap is real; the pilot was under-resolved, not null.
- G_total +0.01566 (|t| 7.20); coverage 106% (both clear 3σ → licensed).
- Trio {int8,bf16,fp16} flat within ±0.0019 → **stop-at-int8 ships**. fp4−nf4 NOT separable
  (t 2.24); all 4-bit-vs-≥8-bit pairs survive Bonferroni at t 7.2–9.5.
- fp8>{nf4,fp4} fires by letter, but fp8−bf16 n.s. (t 1.39) — regularizer premise dead;
  S-D stays closed.
- Tail: top-10% = 45% (subpopulation branch does NOT fire, <50%).
- D2 at n=1024: 6×1024 bitwise placement-identical; determinism repeat 1024/1024 bitwise.

**P-A (Addendum 1) — the pilot factor structure was the shared-outlier artifact:**
- **P-A1 FAILS** (PC1 39% < 50%); **P-A2 FAILS** (ρ −0.05); **P-A4 FAILS** (fp8 flips 6.0 vs
  nf4 16.7 — fp8 routes 8-bit-like). nf4−fp8 deviation corr 0.790 → −0.009 on the disjoint set.
- **P-A3 PARTIAL**: ρ(int8,fp16) = 0.527 replicates (smooth channel real), flip-free clause fails.
- **P-A5 HOLDS**: pilot fragile examples transfer at 5× enrichment (3/6 above the n=1024 cutoff).
- Jaccard(int8,bf16) 0.9955 ≫ (nf4,bf16) 0.9806 — HOLDS.

**O-4 (W_RMS):** ordering = reconstruction chain exactly (no conflict); Addendum-1's
"fp8 near 4-bit" REFUTED (fp8 W_RMS 0.0248, 3.7× closer to int8 than nf4).

**S-B (adapters steer routing):** J(base,adapted) 0.94 ∈ [0.85,0.97] HOLDS; corr(shift, gain)
0.58–0.60 HOLDS (graduation met) — but shift 0.055 > both precision perturbations, so the
"between int8 and nf4" clause FAILS: the adapter is the dominant routing mover.

**D3 rev3:** engagement attested (16/16 evicted, single-slot honored); null+placement bitwise —
certificate self-contained.

**P-C1 (Addendum 3) FAILS:** no shape model reaches <0.10 GB residual on offload peaks.

## 2026-07-05 — Filed, not graded (index)

- **P-B1..P-B4** (cache-lane posture: margin→locality, churn quartiles, lane odds
  40%/15%/70%, re-pin coupling) — `docs/SPECULATIVE_LANES_ADDENDUM_2.md`, timestamped by
  its commit; P-B1/P-B2 await the margins×traces join; P-B3 grades at program end.
- **P-A1..P-A6** (factor structure at n=1024) — `docs/SPECULATIVE_LANES_ADDENDUM_1.md`;
  graded when `scripts/analyze_factor_1024.py` runs post-drain.
- **Amendment §4 branch table** (n=1024 primary, trio resolution, fp8, tail, routing) —
  `docs/NULL_LADDER_1024_AMENDMENT.md`; graded by `scripts/analyze_ladder1024.py`.
- **S-A/S-B/S-C committed predictions + surprise-bet order S-A>S-C>S-B>S-E>S-D** —
  `docs/SPECULATIVE_LANES_PLAN.md` (S-A items superseded by P-A series per Addendum 1).

## 2026-07-05 — Addendum 3 §4 (recorded verbatim) + same-day executor gradings

Reviewer's own gradings (source-read addendum, authored from main_11.zip):

- **RETRACTED (wrong):** "workspace removal → nf4-offload ≈ 1.8 GB." The mechanism it
  assumed does not exist.
- **CORRECTED (overreach):** prior turn's "~85%, RNG mechanism named in a docstring."
  Actual content of the read: one candidate eliminated, none confirmed.
- **CORRECTED (inference vs measurement):** "atomics → non-bitwise" deferred to D2's
  measured bitwise repeat. R9 cuts both ways.
- **STANDS:** the bf16-offload memory inversion and its user-facing claim.

Executor gradings on the same window:

- **P-C1 (Addendum 3 §2) FAILS**: no shape-derived model reaches per-mode residual
  < 0.10 GB against the six offload train peaks (A/B miss +0.63 quantized; C misses
  −0.17) — no mechanism sentence ships (`docs/OFFLOAD_MEMORY_FACTS.md` T4b).
- **T1.0b resolved**: torch 2.8 non-reentrant checkpointing preserves RNG
  unconditionally; HF passes kwargs verbatim; offload.py touches zero generators
  (`docs/LAYOUT_FACTS.md`). Combined with no-dropout-on-path (T1.0) and the bitwise
  certificate (D3), the RNG sub-mechanism is dead three ways.
- **Addendum 3 §1's ~75–80% environment-branch estimate: overtaken by measurement**
  before filing — D3's five trios are bitwise in null AND placement; no one-step
  divergence mechanism exists to apportion.

## 2026-07-05 — Lanes Addendum 1 §2 (recorded verbatim, per Z5)

- **FALSIFIED (n=64):** the example-level independence gloss "the scattered formats
  flip different subsets." Deviations share a dominant common core; nf4 is a
  near-pure readout of it. Expert-level sharing (same experts vs same examples)
  remains OPEN until telemetry.
- **STILL OPEN, not falsified:** the committed prediction corr(|d_i|, flip count) ≈ 0
  within the {int8, bf16, fp16} trio. Under the two-factor reading this is expected
  to HOLD (factor 2 is the flip-free channel). The prior chat grading conflated
  deviation–deviation correlation with |d_i|–flip correlation; this entry corrects it.
- **STANDS:** the SD-clustering observation; the fp8-nearer-nf4-than-int8 expectation
  (now supported at the variance level).

### Z1 post-verification annotation (same day, after the raw-vector rerun)

The addendum's §1 numbers reproduce exactly on the raw vectors. The preregistered
Pearson-vs-Spearman check then **largely confirms the shared-outlier alternative for
factor 1**: scattered-format correlations collapse under ranks (nf4–fp8 0.790→0.145,
nf4–fp16 0.520→0.032) while int8–fp16 survives (0.745→0.636). The "common core" is
substantially a shared fragile-example subpopulation; factor 2 is the distribution-robust
channel. Details: `runs/results/postaudit/factor_structure_n64.md`. Z4 resolved by Z3
(shared compute path; fp16 = minimal weight perturbation, not an activation channel —
`docs/OFFLOAD_MEMORY_FACTS.md`).

## 2026-07-05 — Certificate predictions P1–P4 (`docs/TRAIN_PLACEMENT_CERTIFICATE.md`)

- **P2, P3, P4 CONFIRMED** (placement bitwise-exact, all five trios).
- **P1 WRONG in the informative direction**: the default-kernel one-step null is bitwise;
  the MoE-combine atomics did not produce noise on these shapes.

## 2026-07-05 — Amendment §6 ledger note (adopted rule, recorded verbatim)

No significance forecasts without the instrument's n in hand. The prior forecasts
(75% G clears; 10–15% S9) missed on a knowable instrument parameter, not on physics.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-06T04:36:37Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `e1e509306a1392ea765fb6b7247030cdd11bde4e1c1be3c69c8bbcbb145a017c` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-06T02:10:19Z` `1eabe6aedfaf050152834a38f631bc4f06d1661f3b85a3aefeb56a8a85b33e55`
  - `2026-07-06T02:01:24Z` `abdbb765824bcc92396b7dc401e7a7e645c6e72aa9747ea8060cd8c71a1cf1b0`
  - `2026-07-06T00:36:02Z` `3f7b908c9fececbefdf161bdfde661363339406cc05c77f99c48d97013438fed`
  - `2026-07-06T00:19:00Z` `c1d2c77eacd785d924482fdec33fcab841d843b751139dc64431f57a4152b8e1`
  - `2026-07-05T22:22:31Z` `80f86be8b06dcbb5e0a9d96013f1435ac5daa93b16049ad212006d91461bbf47`
  - `2026-07-05T20:04:03Z` `f23487b74d50e842b37b930f02174c72ecf2846b18ca542b55d4424d56b2e765`
  - `2026-07-05T18:22:04Z` `a49fc7cb9f351c8096608c5ea1bb281e065c237f9d6d8268c055035c7ea9c2b4`
  - `2026-07-05T18:18:28Z` `5ebfa4b4193e3351723ed1aa29578ae2f53de51e7d34fc54bcb075e7aa712024`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[?:?O.#~.0%:~#+?%]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|  ... +o.        |
|  .. oo+oE +     |
| .  +o o=.X *    |
|.  . .o. B.%     |
|.    . .SoB .    |
| o .  oooo.o     |
|. . . oo.o.      |
|     . .+o       |
|        .++      |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info PREDICTION_LEDGER.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify PREDICTION_LEDGER.md.ots PREDICTION_LEDGER.md` succeeds against the on-disk bytes.
- Anchor file: `PREDICTION_LEDGER.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
