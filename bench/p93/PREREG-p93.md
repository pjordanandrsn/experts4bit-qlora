# P93 — does K25 with the select tree at the served precision make the NF4 families' B=16 decode faster on an RTX 5090 without moving their K8? Granite-3.1-3B-A800M and OLMoE-1B-7B, OFF vs ON (registered 2026-10-01, before any run)

Issue: experts4bit-qlora#564. **This is P92's design** (`bench/p92/PREREG-p92.md`), re-read on the route as it now
stands.

**What changed since P92:**
- **P92** (`bench/p92/RESULTS-p92.md`) read K25 in bf16 with the paired codebook lookup:
  - Granite LICENSED (B=16 ×0.937, K8 −0.023 / +0.016 ppl);
  - OLMoE QUALITY_FAIL (c4val1 K8 −0.107 ppl, step ×1.024).

  K25's own GEMM ran within 4 % of the served one.
- **grouped-nf4-gemm K26** (#432) found the codebook lookup is about 80 % of K25's time. An exact select-tree decode is
  bit-identical to it at 0.37–0.38 of the time. That tree is K25's default now (#433).
- **K27** (#435) read K25-tree at the served kernel's weight precision (fp32 weights through TF32 MMA,
  `dot_bf16=False`), at its best plan BLOCK_N 32 / KC 64 / 4 warps / 3 stages. It runs at **0.448 (Granite) and 0.502
  (OLMoE) of the served NF4 GEMM's time**, with an rms error against fp64 equal to the served kernel's (ratio 1.000).
  TF32_PATH by its rule.
- **The route** (`E4B_NF4_GROUPED_SMALLM`; this lane's launch commit) runs K25 at exactly that:
  `_K25_PLAN = {block_n 32, kc 64, warps 4, stages 3, lut "tree", dot_bf16 False}`.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- receipts and ledger rows;
- proven teardown.

## What is the same as P92 (by reference, unchanged)

- **The box, the harness, the families and their licensed envs.** One RTX 5090, any CPU vendor; P91's harness; Granite
  `r12epi` and OLMoE `nf4` at P44's revisions.
- **The arms:** OFF = `E4B_NF4_GROUPED_SMALLM=0`, ON = `=1` (T == 1 included, so K8 reads the kernel).
- **The order:** the premise on the card (the K25 row-exact test, rc 25; K25's contract compiled, rc 23); then, per
  family, B=16 OFF/ON/OFF/ON (draw 1 censused), B=1 OFF/ON (censused), and K8 OFF/ON on wikitext and c4val1.
- **The reducer:** `p93_reduce.py` is `p92_reduce.py` with the lane renamed. The same 14-case self-test and the same
  rule, per family:
  - VOID on missing arms, B=16 draws more than 3 % apart, failed engagement (`_gemm_nf4_grouped_smallm` 2 × layers per
    step ON with no served NF4 kernel, none OFF), or K8 ON bit-equal to OFF;
  - QUALITY_FAIL if the uncalibrated K8 gate fails (|Δppl| ≤ 0.05 per text);
  - LICENSED if B=16 ON/OFF ≤ 0.95;
  - NOT_FASTER otherwise.
- **The registered consequence:** both families LICENSED moves `E4B_NF4_GROUPED_SMALLM` to `auto` (T == 1 too if B=1
  ON/OFF ≤ 1.00 in both). Otherwise the default stays `0`.

## What changes

1. **The kernel pin:** grouped-nf4-gemm at `908a2ca` (main at K27's registration, #434). It carries the tree decode
   (#433), and K27's read (#435) changes only docs and receipts. The tripwire refuses a kernel package without the
   tree, or a route whose `_K25_PLAN` is not the TF32 plan above.
2. Nothing else.

## Predictions (written before the data)

- **B=16 ON/OFF:** Granite **0.55–0.75**, OLMoE **0.55–0.75**. Both LICENSED.
  - In P92's census the served GEMM was 6.60 (Granite) and 10.39 (OLMoE) ms/step, on a slower box. At K27's 0.45 /
    0.50 it becomes about 3.0 / 5.2 ms. The lean glue saves its 0.35 / 0.17 again. So the steps fall from about 10.5 /
    14.4 to about 6.6 / 9.0 ms: ×0.62 / ×0.63, before host spread.
- **B=1 ON/OFF:** **0.70–0.95** in both. At one row per expert the tree's cheap decode should beat the scalar decode
  GEMV, which P92 read at ×0.97 / ×1.07 against the paired lookup.
- **K8:** **|Δppl| < 0.02** on both texts in both families, the same precision class as the served route.
  - P92's bf16 moves (up to 0.107 on OLMoE c4val1) came from bf16 weight rounding, which TF32 does not do.
  - At T == 1, OFF runs an fp32 FMA GEMV and ON a TF32 MMA. TF32 rounds the weight to a 10-bit mantissa, well below
    NF4's own error.
- **The default moves to `auto`.** T > 1 only, unless B=1 reads ≤ 1.00 in both families, which I expect it will.

## Box and cost

- **`p93-5090-<n>`:** one RTX 5090, **guard 1.0 h at ≤ $0.75/h (≤ $0.75)**, so there is no proving rental. P92's run
  took 35 minutes.
- **Lane ceiling $1.50; hard stop $2.00.**

## Rehearsal

The whole runner ran on the NAS RTX A2000 (sm_86) from e4b `826c733`, this branch before this section, with
grouped-nf4-gemm at the pin. It ran with `P93_GPU_CLASS=A2000 P93_MIN_DISK_GB=20`, so it was marked REHEARSAL.
- **Up to the arms, it held:**
  - install and the new tripwire (the tree decode present; `_K25_PLAN` the TF32 plan);
  - the reducer's self-test (14 cases);
  - the premise: row-exact 4/4, K25's contract compiled 30/30;
  - the families fetched and baked.
- **Every arm stopped** at the fp8 paged-KV append (`fp8e4nv`, which sm_86 cannot compile), as in P91's and P92's
  rehearsals.
- The reducer then read VOID for both families, with every arm missing, as it must.
- No time is quoted (the A2000 is correctness-only).

Amendments, dated, go below this line before any data is read.
