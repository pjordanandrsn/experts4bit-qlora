# TC1b — Qwen3-30B-A3B, one RTX 5090: the long curve re-asked with matched init and matched precision (200 steps, e4b vs Unsloth), the as-shipped e4b curve beside it, and the tp2 anchor pair (registered 2026-10-01, before any box; family token `qwen3curve` of the TC1 harness)

Issue: experts4bit-qlora#835. Lineage: P38 (`bench/h2h-20260905/p38/`): at tp2's fixture the 200-step held-out curves
separated in Unsloth's favour (0.2713 vs 0.2881) while e4b led at every eval before step 80; e4b's adapters on that
row were part bf16 ("torch.bfloat16, torch.float32"), Unsloth's fp32; the two inits differed by construction. The
causes were listed as candidates and never tested. tp4's amendment 2 cut the field-recipe instrument to N = 20 and 8
held-out rows, so no field-recipe curve beyond 20 steps exists on this family.

## Claim under test

With the init, the adapter precision, the optimizer call, the tokens and the schedule matched, do e4b's fused path and
Unsloth's 4-bit MoE path produce the SAME held-out curve over 200 optimizer steps of the field recipe — and does the
as-shipped e4b configuration (bf16 expert adapters, N(0, 1/r) init) reproduce P38's shape (fast early descent, then a
plateau above the matched curve)?

## Fixture

TC1's (tp4's field recipe) with N = **200**, eval every **40** steps (0, 40, 80, 120, 160, 200) on the first **16**
held-out rows (paired across arms; a wider instrument than TC1's 8 rows because the reading is a curve); the linear
schedule's decay runs to step 200 (transformers' formula with warm-up 5), as the recipe defines it for `max_steps`.
Same model pin; the training rows are byte-identical to TC1's (the box asserts a train-only sha, `train_sha`, because the
tokens file's own sha also covers the eval rows, which differ: 16 here against TC1's 8).

## Arms, in this order

1. `e4b/fused_attn4_m_200` — matched init, fp32 adapters, `enable_fast_train(dgrad=True)` + attention 4-bit.
2. `unsloth/ckpt_unsloth_m_200` — matched init, fp32 adapters, Unsloth 2026.9.14 at its own pins in `venv-unsloth` (torch 2.12.1+cu130, `UNSLOTH_MOE_BACKEND=grouped_mm`, as TC1's comparator).
3. `e4b/fused_attn4_shipped_200` — as built by the loader (bf16 expert adapters, N(0, 1/r) init).
4. Anchor pair at tp2/P38's fixture (clinical text, seq 512, batch 1 x accum 1, r 8 / alpha 16, lr 1e-4, torch AdamW
   wd 0.01 constant, seed 0, N 60, 8 held-out rows as tp4 ran it): `e4b/fused_attn4_p38` and `unsloth/ckpt_unsloth_p38`
   — native init and native precision; the e4b arm is byte-for-byte tp4's; the Unsloth arm runs in `venv-unsloth` (cu130,
   grouped_mm) and a second `unsloth/ckpt_unsloth_p38_t28` runs in `venv-unsloth-t28` with the default backend, which IS
   byte-for-byte tp4's arm (8 held-out rows, as tp4 RAN its anchor — amendment 3 there; the one per-arm environment
   difference is `OMP_NUM_THREADS` set to the physical core count, phase 3 of TC1), so the t28 ratio reads against tp2's
   1.457 and P38's 1.413 — tp4's own anchor ratio is not in this tree (its reread names receipts by sha only), so P4 reads
   against tp2's — and the cu130 ratio is the same fixture on the comparator's intended backend. Every receipt and stub
   carries a verbatim provenance `--note` naming its sub-fixture.
5. The tokens-per-step scaling pair (where the gap comes from): `e4b/fused_attn4_m_t1` and `unsloth/ckpt_unsloth_m_t1`
   at seq 2048, micro-batch 1 x accum 1, N 20, matched init and precision — the middle point between the anchor's
   512 tokens/step and the field recipe's 16,384: with tp2's 1.46 at 512 and tp4's 4.49 at 16,384, a ratio that grows
   with tokens per step says the competitor's per-expert dequant-and-matmul cost scales with routed tokens x experts
   while the grouped kernel amortises it; a flat ratio says it does not. Registered as the causal reading for TC1's
   position, and as the counterexample search in the small-batch direction.
6. The rank pair (where the delta GEMMs grow): `e4b/fused_attn4_m_r64` and `unsloth/ckpt_unsloth_m_r64` at r 64 /
   alpha 64, N 20, matched init and precision, the field recipe otherwise — the direction in which e4b's padded
   grouped-LoRA delta pads more and the competitor's two extra grouped GEMMs grow: a registered counterexample search.
Every eval records per-row losses; the curve reading uses paired mean +- SE over the 16 rows at each eval step.
Alarms: e4b 200-step arms 4,800 s each (expected ~40 min: 200 x ~8 s + 6 evals of 16 rows), Unsloth 9,000 s (expected
~115 min: 200 x ~29 s + evals), anchor arms 1,800 s each. Order puts the matched pair first so a deadline cannot eat it.

## Validity and verdicts

TC1's (`bench/tc1/TC1-PREREG.md`), applied by the same reducer: (tp4's predicates, matched-init completeness, fp32 adapters, step-0 within 0.01 of each other for arms 1-2 (SAME-BYTES-CLASS), one matched_init_sha across the matched arms,
C1, tokens sha, trainable count 642,514,944 on arms 1-3 and 321,257,472 on the anchor arms). Verdict column as TC1.

## Readings

- **Curve equivalence** (arms 1 vs 2): at every eval step the paired held-out mean over the 16 rows, +- SE, and the paired
  |delta|; EQUIVALENT-AT-EVERY-EVAL iff every paired |delta| <= 0.02 (this lane has no in-draw reference arm, so the band is
  the fixed 0.02 and the TC1 floor is quoted beside it), else DIVERGENT with the first step of divergence; the curve reading is
  the largest |delta| over the six evals and its sign at step 200.
- **The plateau test** (arm 3 vs arm 1): held-out at 200 of the as-shipped arm minus the matched arm; a positive gap
  >= 0.01 nats reproduces P38's shape with the init/precision confound now isolated on e4b's own side.
- **Time to a held-out target** (informational, all three arms): wall-clock to reach the matched pair's step-200 loss
  + 0.02, read from the eval grid (the first eval at or below the target).
- Speed per step over steps 11..200 for arms 1-3 (not a position: TC1 owns positions; reported as a check that the
  20-step median travels to 200 steps within 10 %).
- Anchor: the p38 pair's ratio vs tp2 1.457 / P38 1.413 / tp4's anchor row, +-10 % AGREES, else a finding.

## Predictions

- P1 arms 1 and 2 are EQUIVALENT at every eval (max paired |delta| <= 0.02); refuted -> DIVERGENT, the sign and the
  step of first divergence are the finding, and no training position on this family is quoted until TC4 explains it.
- P2 arm 3 ends >= 0.01 nats ABOVE arm 1 at step 200 while being at or below it at step 40 (P38's shape, now
  attributed to init scale and/or bf16 adapters on e4b's side). Refuted if arm 3 ends within 0.01 or below.
- P3 the 20-step s/step medians (TC1) and the 200-step medians agree within 10 % per arm.
- P4 the t28 anchor ratio is within +-10 % of tp2's 1.457 (tp4's anchor ratio is not in this tree); the cu130 anchor ratio is
  reported beside it as a new measurement.

## Decision rules

- P1 holds -> the equivalence claim is carried as a 200-step curve, not a 20-step point; TC1's position stands.
- P1 refuted -> TC4 (causal bisection: adapter precision alone, init alone, checkpointing mode, attention kernel)
  before any quote.
- P2 holds -> `load_moe_4bit_streaming`'s adapter defaults become an open item (fp32 expert adapters and PEFT's init
  as defaults, or at least documented), filed, not changed under this lane.

## Budget and stop rules

One RTX 5090, Vast verified-secure, ceiling $0.69/h, guard 5.5 h, estimate $3.80 (< $15). Same pre-flight and STOP rules
as TC1. The 200-step Unsloth arm is the long pole; if a draw cannot hold it before the deadline it is `not_run`
host-limited and the lane is redrawn once with the anchor pair dropped (an amendment, dated).

## Amendments

(none yet)
