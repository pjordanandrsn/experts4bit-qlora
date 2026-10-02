# P95 — K8's spread across arithmetics of equal per-GEMM error: the scalar GEMV, the served NF4 M-tile and K25 at T == 1 on disjoint windows of each text, on Granite-3.1-3B-A800M and OLMoE-1B-7B on one RTX 5090 (registered 2026-10-02, before any run)

Issue: experts4bit-qlora#564.

**Why.**
- **P94** (`bench/p94/RESULTS-p94.md`, #847) read K8 at T == 1 under three arithmetics, on one window of each text:
  - g, the scalar GEMV;
  - m, the served NF4 M-tile kernel;
  - t, K25 with the select tree through TF32.
- **The c4val1 pairs differed by 0.013 to 0.168 ppl.** Production's own two arithmetics, g at T == 1 and m at T > 1,
  read −0.078 apart on Granite c4val1. On wikitext every pair was within 0.026.
- **grouped-nf4-gemm's K27** read the per-GEMM rms error of t and m against fp64 equal to four digits.
- **The question.** With one draw per arithmetic, P94 cannot say whether c4val1's K8 has a realization spread of
  that size across equal-error arithmetics (the gate below the instrument's resolution), or whether K25 is biased
  on OLMoE.
- **This lane measures that spread.** It licenses nothing. P94's verdict stands under its own rule, and the 0.05
  threshold does not move.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

- **Box, software, families:** P94's. One RTX 5090.
  - e4b at the launch commit; the arms' code is unchanged since P94's `966a95e`.
  - grouped-nf4-gemm at P94's `908a2ca`, so window 0 can reproduce P94 bit for bit.
  - Granite `r12epi` and OLMoE `nf4` at P44's revisions, each on its licensed env.
- **The three arithmetics at T == 1:** P94's g, m and t, set by the same environment.
- **The windows.** Window k is the K8 corpus from token k × 4096: `--prompt-offset k*4096 --prompt-span 2600`.
  - Each window holds a 512-token prompt and 2048 scored tokens, and windows do not overlap.
  - **Window 0** is P94's window, run with P94's exact command line (no offset flags). It is the reproduction control.
  - **Fresh windows:** c4val1 1–8, wikitext 1–4.
- **The premise, on the card, before anything is fetched:**
  - P94's two row-exact files (rc 25);
  - grouped-nf4-gemm's K25 contract, compiled (rc 23).
- **The order:**
  1. Both families are fetched and baked.
  2. Each family's m arm runs at B=1, censused, for engagement.
  3. Then window-major: for each window k, Granite then OLMoE, c4val1 then wikitext (while k ≤ 4), arms g, m, t.
     A deadline therefore trims the highest windows of both families evenly.

**The reducer** (`p95_reduce.py`, 12-case self-test). Per family and text it uses the complete fresh windows (all three
arms present).
- **The per-window spread.** The deltas of the two equal-error pairs, m − g and t − m, each give a sample SD. The
  instrument's per-window spread is **σ = max(SD(m − g), SD(t − m))**.
- **VOID** if any of these holds:
  - the m arm's engagement fails (`_gemm_nf4_grouped` exactly 2 × layers per step, no `_gemv_nf4*`);
  - a window-0 arm is missing;
  - there are fewer than 6 complete fresh windows on c4val1, or 3 on wikitext;
  - a window's arms scored different text;
  - two windows scored the same text;
  - m or t is bit-equal to g in every complete fresh window of a text.
- **RESOLVED** if σ ≤ 0.025 on every text in both families. A single-window 0.05 gate then falsely fails an unbiased
  equal-error change at most about 5 % of the time (2σ).
- **UNDER_RESOLVED** otherwise. For each family and text above 0.025, the reducer reports how many windows a
  windowed-mean gate needs for the same ~5 % bound: **W = ⌈(σ / 0.025)²⌉**.
- **Reported, not gated:**
  - each pair's mean, SD and SE over the fresh windows;
  - whether t − m's mean is a consistent shift (|mean| > t₀.₉₇₅(n − 1) · SE);
  - whether window 0 reproduces P94's twelve values exactly.

**The registered consequence.**
- **RESOLVED:** the single-window K8 gate stands. P94's QUALITY_FAIL then reads as K25's own shift, beyond the
  instrument's spread, and the K25 default question closes for these two families at this plan.
- **UNDER_RESOLVED:** P94's verdict still stands; this lane licenses nothing. Any later quality lane on these families,
  including one that asks again about K25's default, registers a **windowed K8 gate**:
  - |mean delta over W fresh windows| ≤ 0.05 on every text;
  - W at least this lane's `windows_needed` for that family and text;
  - windows disjoint from P94's and P95's (k ≥ 9);
  - its rule fixed before its data.
- **VOID:** nothing moves.

## Predictions (written before the data)

- **Window 0 reproduces P94 exactly:** a deterministic instrument on the same card class with the same pins.
- **c4val1:** σ between 0.04 and 0.10 in both families, so UNDER_RESOLVED, with `windows_needed` between 3 and 16.
- **wikitext:** σ ≤ 0.025 in both families.
- **Granite c4val1 m − g:** a mean near 0. P94's −0.078 is one draw.
- **OLMoE c4val1 t − m:** no consistent shift. This is the least certain prediction.

## Box and cost

- **`p95-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P95_PROVE=1` runs:
  - the refusals;
  - the install with its tripwire;
  - the reducer's self-test;
  - the premise;
  - an HF CDN egress probe.

  It loads no model.
- **`p95-5090-<n>`:** one RTX 5090, after a passing proof. **Guard 2.0 h at ≤ $0.75/h (≤ $1.50).** About 100
  minutes:
  - install and premise ~5;
  - fetch and bake ~5;
  - two censuses ~3;
  - 84 K8 arms at about 63 s each.
- **Lane ceiling $3.00; hard stop $4.00.**

## Rehearsal

To be written from the A2000 rehearsal before this page merges.

Amendments, dated, go below this line before any data is read.
