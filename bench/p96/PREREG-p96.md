# P96 — K25's default, asked again with P95's windowed K8 gate: K25 against the served NF4 M-tile at T == 1 on fresh windows of each text, on Granite-3.1-3B-A800M and OLMoE-1B-7B on one RTX 5090 (registered 2026-10-02, before any run)

Issue: experts4bit-qlora#564.

**Why.**
- **P93** (#837) measured the served-precision K25 route at B=16 ×0.594 (Granite) and ×0.598 (OLMoE).
- **P94** (#847) read its K8 against the arithmetic it would replace, the served NF4 M-tile. On one window per text it
  read QUALITY_FAIL: c4val1 +0.102 / +0.168.
- **P95** (#866) measured that one window cannot resolve 0.05 on these families.
  - K8's per-window spread across arithmetics of equal per-GEMM error is 0.054 / 0.055 on c4val1 and 0.026 / 0.030
    on wikitext (UNDER_RESOLVED).
  - It named the gate a later quality lane registers: |mean delta over W fresh windows| ≤ 0.05 on every text, with
    W ≥ 5 on c4val1 and W ≥ 2 on wikitext, on windows disjoint from P94's and P95's.
  - Over P95's fresh windows, K25 showed no consistent shift. P95 registered that it licenses nothing.
- **This lane** asks the default question again under that gate, on new windows, with its rule fixed here. P94's and
  P95's verdicts stand under their own rules.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

- **Box, software, families:** P95's. One RTX 5090.
  - e4b at the launch commit; the arms' code is unchanged since P94.
  - grouped-nf4-gemm at `908a2ca`.
  - Granite `r12epi` and OLMoE `nf4` at P44's revisions, each on its licensed env.
- **Two arithmetics at T == 1:**
  - **m:** the served NF4 M-tile kernel (`E4B_NF4_GROUPED_SMALLM=0`, `E4B_NF4_T1_DEVICE_GROUPING=1`). This is what the
    `auto` default replaces above T == 1.
  - **t:** K25, the select tree through TF32 (`E4B_NF4_GROUPED_SMALLM=1`).
- **The windows.** Window k is the K8 corpus from token k × 4096 (`--prompt-offset k*4096 --prompt-span 2600`): c4val1
  9–16 (8 windows) and wikitext 9–12 (4 windows). P94 used window 0 and P95 used 0–8. These counts are above P95's
  minimums (5 and 2), which lowers the chance of falsely failing an unbiased change.
- **The premise, on the card, before anything is fetched:**
  - P94's two row-exact files (rc 25);
  - grouped-nf4-gemm's K25 contract, compiled (rc 23).
- **The order:**
  1. Both families are fetched and baked.
  2. Each family's m and t arms run at B=1, censused, for engagement.
  3. Then window-major: for each window, Granite then OLMoE, c4val1 then wikitext (while k ≤ 12), arms m, t.

**The reducer** (`p96_reduce.py`, 12-case self-test).
- **VOID** if any of these holds:
  - an engagement census fails. m must run `_gemm_nf4_grouped` exactly 2 × layers per step; t must run
    `_gemm_nf4_grouped_smallm` exactly 2 × layers per step; neither may run the other's kernel or `_gemv_nf4*`;
  - a text has fewer complete windows than P95's minimum (c4val1 5, wikitext 2);
  - a window's arms scored different text;
  - two windows scored the same text;
  - t is bit-equal to m in every complete window of a text.
- **LICENSED** if |mean(t − m)| ≤ 0.05 ppl over the complete windows, on every text, in both families.
- **QUALITY_FAIL** otherwise.
- **Reported, not gated:** each text's per-window t − m with its mean, SD and SE, and the SD beside P95's σ for that
  text.

**The registered consequence.**
- **LICENSED**, with P93's speed (both families B=16 ≤ 0.95, already read): `E4B_NF4_GROUPED_SMALLM` defaults to
  `auto`. That covers rows above T == 1 only; T == 1 stays on the GEMV.
- **QUALITY_FAIL:** the default stays `0`.
- **VOID:** nothing moves.

## Predictions (written before the data)

- **LICENSED.** |mean t − m| < 0.04 on every text in both families. Over P95's fresh windows the means were −0.019 /
  +0.028 (c4val1, SE 0.019) and +0.002 / −0.010 (wikitext).
- **This lane's per-window SD of t − m** lands within a factor of 1.5 of P95's σ for each text.
- **The engagement censuses** read 64 / 32 calls per step for each arm's kernel, as P93 and P94 read them.

## Box and cost

- **`p96-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P96_PROVE=1` runs:
  - the refusals;
  - the install with its tripwire;
  - the reducer's self-test;
  - the premise;
  - an HF CDN egress probe.

  It loads no model.
- **`p96-5090-<n>`:** one RTX 5090, after a passing proof. **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).** About 60
  minutes:
  - install and premise ~5;
  - fetch and bake ~5;
  - four censuses ~4;
  - 48 K8 arms at 45–64 s each.

  The runner stops starting arms 20 minutes before the deadline.
- **Lane ceiling $2.50; hard stop $3.50.**

## Rehearsal

Run on the NAS RTX A2000 (sm_86) from e4b `386d46b`, this branch before this section, with grouped-nf4-gemm at the pin.
It used `P96_GPU_CLASS=A2000 P96_MIN_DISK_GB=20`, so it was marked REHEARSAL. No time is quoted.
- **The proving run** (`P96_PROVE=1`) held end to end, rc 0:
  - install and the tripwire;
  - the reducer's self-test (12 cases);
  - the premise: row-exact 6/6, K25's contract compiled 30/30;
  - the HF CDN probe;
  - `PROVED`.
- **The windows, checked directly** through `step_decomp._k8_window` on both families' tokenizers. Every window holds
  the 2,561 tokens it needs (2,600-token slices), and the 24 scored-window digests are distinct.
- **The full arm path** (fetch, bake, censuses, the window-major K8 loop) is P95's.
  - It was rehearsed there, and it ran for real in `p95-5090-1`.
  - P96 changes three things in it: the arm list (m, t), the window lists, and a second engagement census.
    `tests/test_p96_staged_pin.py` pins all three.

Amendments, dated, go below this line before any data is read.
