# RESULTS — the slot-value curve: REFUTED on an accepted box

Scored under [PREREG-slotvalue.md](PREREG-slotvalue.md) (ladder amended
pre-gate, disclosed there) by the committed
[slot_verdict.py](slot_verdict.py); receipts in [receipts/](receipts/).
Box: EPYC 9655 (48 phys cores, B_dram 421.5 GB/s triad, grouped scatter
121.6% of triad) + RTX 5090, torch 2.13.0+cu130, transformers 5.15.1,
pool 32 / torch-intraop 8. Cycle cost ≈ $1.00, box destroyed, zero
instances verified.

## Verdict

**REFUTED — and scoreably so.** The A/A gate ACCEPTED (median pass
noise 2.2%, worst 3.2%), the spoiler passed (full-range ΔT 10.36 ms >
3× the 2.13 ms pass spread), and the bars still failed: B1 fails 2/4
brackets, B2 fails 3/4, B3 fails outright. The claim "every slot
removes its expert's P(touched) × 58 µs/step … ~17–35 µs/slot at the
tail" does not describe this operating point.

## The sweep (both passes; uniques identical across passes — routing is
deterministic over the fixed prompt set)

| vram_gb | slots | DRAM experts | uniq/step | dram ms (p1/p2) | step ms |
|---|---|---|---|---|---|
| 7.0 | 2831 | 3313 | 308.9 | 19.54 / 19.97 | ~215 |
| 9.0 | 3640 | 2504 | 113.8 | 14.69 / 14.75 | ~208 |
| 10.0 | 4045 | 2099 | 62.1 | 12.01 / 11.82 | ~205 |
| 10.75 | 4348 | 1796 | 49.6 | 10.48 / 10.15 | ~201 |
| 11.5 | 4652 | 1492 | 40.4 | 9.53 / 9.25 | ~200 |

| bracket | +slots | ΔU pred | ΔU meas | B1 | ΔT pred | ΔT meas | B2 | µs/slot |
|---|---|---|---|---|---|---|---|---|
| 7→9 | 809 | 248.8 | 195.2 | FAIL | 11.32 | 5.03 | FAIL | 6.2 |
| 9→10 | 405 | 55.3 | 51.7 | pass | 3.00 | 2.81 | pass | 6.9 |
| 10→10.75 | 303 | 20.9 | 12.5 | FAIL | 0.72 | 1.60 | FAIL | 5.3 |
| 10.75→11.5 | 304 | 9.3 | 9.2 | pass | 0.53 | 0.92 | FAIL | 3.0 |

B3: tail value **3.0 µs/slot** vs the widened band [12, 45] → FAIL.

## What the receipts say broke, mechanism by mechanism

1. **The P(touched) independence model over-predicts.**
   `P = 1 − (1 − p)^16` assumes the step's 16 tokens route
   independently. Measured: the deep bracket (7→9) removes 195 uniques
   where independence predicts 249 (−22%), and 10→10.75 removes 12.5 vs
   20.9 (−40%). Within-step co-routing — the same correlation that makes
   batch amortization beat B·k — concentrates draws on fewer distinct
   experts than independence allows. (Cross-request independence, which
   the routing-overlap pre-measurement certified at R ≈ 1.0 on other
   models, is a different quantity and is not contradicted.)
2. **58 µs/unique is an average, not a marginal — and the marginal has
   two regimes.** Dense regime (7→9, ~4+ uniques/layer-call): the
   measured marginal is ~26 µs/unique — on this 421 GB/s host the
   per-unique slope is far below the reference average. Sparse regime
   (top brackets, <1 unique/layer-call): ~100–128 µs/unique — removing
   a tail unique often removes a whole layer call's fixed cost with it,
   so the marginal EXCEEDS the average there. A single flat constant
   cannot carry both; the claim's arithmetic (P × 58) fails in opposite
   directions at opposite ends of the curve.
3. **The measured slot-value curve is 3–7 µs/slot** across margins
   2831→4652 — a factor 3–6 below the quoted band. The band's
   derivation multiplied a tail P (0.3–0.6) by the flat 58; both inputs
   are wrong at this operating point (the achieved margin P runs lower,
   and the conversion is regime-dependent).
4. **The balance knee caps the payoff.** With the measured routing
   profile and this host's calibration, the solver goes balance-bound at
   **4712 slots (~11.7 GB)**: above the knee it declines additional VRAM
   for experts entirely. The top sweep point (4652) sits 60 slots below
   the knee.

## The FP8 payoff, honestly priced (reported, not scored)

At the operating shape (48 layers, 4 KV heads, hd 128, batch 16, 568
tokens/seq): bf16 KV = 0.893 GB, FP8 (k_groups=4) = 0.475 GB → freed
**0.419 GB ≈ 157 slots** at bpe 2.65 MB. Valued on the measured curve
(~3–5 µs/slot near the margin): **~0.5–0.8 ms/step** — and only ~60 of
those slots are usable before the solver's balance knee, worth
**~0.2 ms/step** at the solver's own operating point. The quoted band
would have promised 2.7–5.5 ms. Longer contexts free proportionally
more KV but cannot move the knee — the knee is a property of the cost
model, not of VRAM supply.

## Disposition (per the registered hard stop)

The slot-growth pricing is **closed as stated**: no FP8 payoff claim may
cite P × 58 or 17–35 µs/slot. What a future registration would need,
in order: (a) a within-step co-routing model replacing token
independence (the measured ΔU table here is its calibration data);
(b) regime-split conversion constants (dense-slope + fixed-cost terms —
the cell model's own H_cell form, applied at serving shapes) replacing
the flat 58; (c) re-derivation on the reference host class, since this
box's 421 GB/s DRAM halves the dense-regime slope. The committed sweep
and verdict instruments run unchanged for that work. The b16close
"-14% to close G8 B=16" arithmetic is unaffected — it never priced
slots; but any plan that counted slot growth toward it inherits this
refutation.
