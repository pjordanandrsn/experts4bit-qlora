# ADDENDUM 4 — S-C's SELECTOR, AS LANE P65 READ IT (2026-09-24)

```
status:   selector addendum written by lane P65's decision rule (bench/p65/P65-PREREG.md,
          "Decision rule"), after the read. It adds to S-C; it edits nothing in
          SPECULATIVE_LANES_PLAN.md, Addenda 1-3 or NEXT_CAMPAIGN_LANES.md, which are
          OpenTimestamps-anchored.
cites:    SPECULATIVE_LANES_PLAN.md §S-C (per-expert bit-width from fragility)
          NEXT_CAMPAIGN_LANES.md, Lane N3 (the S-C mixed-precision cell, half-licensed)
          bench/p65/RESULTS-p65.md (the read) and bench/p65/receipts/ (its census)
          bench/p44/RESULTS-p44.md (P44-a: the per-expert premise, Granite holds, Mixtral not)
          experts4bit-qlora#710
clock:    none. Nothing here is registered to run. The cell below needs a serving
          prerequisite that does not exist yet (§3), then its own pre-registration.
```

## 1. What P65 decided, per family

Lane P65 asked two questions of three families' NF4-served experts. First, which within-layer rankings of the experts
survive a change of calibration text (wikitext-2 train → C4 validation shard 1), measured against the same text's own
resampling. Second, whether activation entropy, Colla-Q's proxy ρ, earns a place in S-C's selector.

A selector is written only where the per-expert premise holds: P44's P3, that ≤ 10 % of experts carry half the
activation-weighted error.

| family | premise | selector | why |
|---|---|---|---|
| **Granite-3.1-3B-A800M-instruct** | holds (P44-a P3 0.0992, GPTQ, 32/32 layers) | **TWO_ARMS** | `rel_act` 0.932 / penalty 0.043, `rho_x_err` 0.910 / 0.063 and entropy 0.882 / 0.086 all survive; entropy is not `rel_act` again (within-layer ρ −0.000 / −0.055) |
| OLMoE-1B-7B-0924-Instruct | no (RTN tail 0.212 on c4val1) | NOT_WRITTEN | recorded: `rel_act` and `rho_x_err` survive; entropy (penalty 0.164) and routing frequency (0.380 / 0.556) do not |
| Mixtral-8x7B-Instruct-v0.1, 16 layers | no (P44-a P3 0.2266; this census 0.2344) | NOT_WRITTEN | recorded: only entropy survives (0.789 / 0.138); `rel_act`, `rho_x_err` and frequency do not |

Numbers are from `bench/p65/receipts/p65_verdicts.json`. A signal survives iff its cross-text rank correlation is ≥ 0.5
and its domain penalty is ≤ 0.15.

**Entropy earns a place on Granite, and only as a weight.** It enters as the ρ in `rho_x_err = ρ · rel_act²`, Colla-Q's
objective `L = ρ‖e − ẽ‖²` in relative form. It does not replace the error metric, and on this family no read here
says it beats it.

## 2. The Granite selector (TWO_ARMS)

**Scores, per (layer, expert):**
- **Arm A, `rel_act`.** `sqrt(rel_gu² + rel_dn²)`: the RTN relative errors of the two projections, activation-weighted
  by the down-projection Hessians. This is P44-a's field.
- **Arm B, `rho_x_err`.** `ρ · rel_act²`, where ρ = σ²_within / σ²_total of the expert's output `y = W_dn h`. This is
  Colla-Q's Eq. 5–7, as `bench/p65/expert_entropy.py` computes it.
- **Baseline arm, `freq`.** Routed token-slots, through `hot_sets_from_profile`, the residency engine's own ranking. On
  Granite it survives the text change (0.891), so it is a fair baseline, not a straw one.
- **Null arm (recommended).** A seeded uniform draw of the same size per layer. It tells "any promotion helps" apart
  from "this ranking helps".

**Selection.** Within each MoE layer, promote the top `q` of the layer's experts by the arm's score. Rank on the
calibration text the cell registers. Per P65, switching between wikitext and C4 costs arms A and B at most
0.063 in rank correlation beyond what resampling the same text already costs.
- **`q` = 10 %** (4 of Granite's 40 per layer), P44-a's own tail: 126 of 1,270 routed experts (9.92 %) carry 50.3 % of
  the error.
- **Only experts with ≥ 32 routed rows are ranked,** P65's and the recipe's threshold. Others keep the base format.

**Matched bytes.** Every arm promotes the same number of experts per layer. Granite's experts share one shape, so every
arm spends the same bytes, and the comparison is at equal memory by construction.

**Outcome.** Arm A against Arm B against the baseline (and the null), each promoted set held at the higher format with
the rest at the base format:
- **Primary:** the two-text K8 (wikitext and c4val1) of the served model against the uniform base-format control, under
  the family's registered K8 floors.
- **Where K8 cannot read,** KL against the same control, per the project's KL rules (a measured floor and the factor-2
  rule, METHODOLOGY §13.1).
- **The comparison that decides:** A against B at matched bytes, and each against the baseline.
- **S-C's original graduation prediction stands** as the cell's quality bar: the promoted cell recovers ≥ 50 % of the
  certified gap at ≤ 15 % of the higher format's memory premium.

**What P65 already rules out.** A selector ranked on one text and served on another is not undermined on Granite by
ranking drift: arms A, B and the baseline each keep a cross-text rank correlation of at least 0.89. Whether the arms differ in *outcome* is the
cell's question, and P65 does not answer it.

## 3. The prerequisite, and the order of operations

1. **Serving two formats in one layer.** Today the store is one scheme per layer (`docs/STORAGE-MODES.md`), so a
   promoted-expert cell cannot be served as specified. The smallest honest form is two stores per layer (base and
   promoted) with the dispatch splitting rows by expert id. Its exactness and speed need their own gate before any
   quality cell rides on it. The alternative is a quality-only instrument that dequantises the promoted experts
   through the prefill branch at decode, in the way `E4B_INT4_DECODE_A16` does for P64. That instrument makes no speed
   claim.
2. **N3's flip-attribution join** (CPU, data on disk) is independent of this addendum and stays where
   `NEXT_CAMPAIGN_LANES.md` puts it. Its concentration test decides whether S-C's fragility map has a tail at all on
   the families it covers. P65's selector decides how to rank inside that tail on Granite.
3. **Then a P-numbered pre-registration of the Granite cell**, with the arms above, their predictions and a proving
   rental.

## 4. What the other two families contribute (recorded, not selectors)

- **Colla-Q's comparative claim replicates on OLMoE and Mixtral.** The claim is that an entropy ranking survives a
  calibration-domain change better than a routing-statistics ranking. Paired, over layers:
  - OLMoE: +0.421 [0.329, 0.501];
  - Mixtral: +0.479 [0.369, 0.588]. Mixtral is Colla-Q's own model.
  - It is unresolved on Granite, where frequency is itself stable (−0.009 [−0.044, 0.024]).
- **The metric they report cannot show this.** Their per-layer cosine reads 0.9995–0.9999 on all three families while
  the rank correlations are 0.79–0.88. That is P65's P5.
- **Mixtral is the ENTROPY_ONLY pattern without a premise.** Entropy is the only one of the four signals that survives
  the text change there. If a future census gives Mixtral a per-expert premise, that branch applies: the addendum it
  writes must first show that entropy predicts the serving text's error.
- **Routing-frequency hot sets do not transfer on OLMoE or Mixtral.** `hot_sets_from_profile` fed one text's profile
  agrees with the other text's hot sets at Jaccard 0.141 (OLMoE, top-6 of 64) and 0.125 (Mixtral, top-1 of 8, exactly
  chance). Granite reads 0.663. This bears on residency profiles calibrated on one text. It is a ranking fact, not a
  hit-rate measurement.

## 5. Prediction ledger for this addendum

No new prediction is committed here. The cell's predictions belong to its own pre-registration (§3.3), written before
its data. P65's own predictions and their grading are in `bench/p65/RESULTS-p65.md`. All held except P1 on Mixtral,
refuted: `rel_act`'s domain penalty there is 0.159 against the 0.15 survival line.
