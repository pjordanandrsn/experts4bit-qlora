# P94 — results: **QUALITY_FAIL in both families**. Against the arithmetic it would replace (the served NF4 M-tile kernel), K25 moves c4val1 K8 by +0.102 ppl (Granite) and +0.168 (OLMoE). The served M-tile itself sits 0.078 from the GEMV on Granite c4val1

Registration: `bench/p94/PREREG-p94.md` (#841, `966a95e`). Issue: #564. The instrument is #838
(`E4B_NF4_T1_DEVICE_GROUPING=1`). grouped-nf4-gemm is pinned at `908a2ca`.

**Verdict by `p94_reduce.py`: `QUALITY_FAIL` (granite FAIL, olmoe FAIL).** Under the registered consequence
`E4B_NF4_GROUPED_SMALLM` stays `0`. The rule is applied as registered; the gate does not move after this failure.

K8 ppl at T == 1 under three arithmetics (2048 steps, eager B=1, `--no-fuse-qkv`):

| family, text | g: scalar GEMV | m: served M-tile | t: K25 TF32 tree | m − g | t − g | **t − m (gated)** |
|---|---:|---:|---:|---:|---:|---:|
| Granite `r12epi`, wikitext | 5.36059 | 5.35515 | 5.33507 | −0.00544 | −0.02552 | −0.02008 |
| Granite `r12epi`, c4val1 | 11.57477 | 11.49663 | 11.59831 | **−0.07814** | +0.02353 | **+0.10167** |
| OLMoE `nf4`, wikitext | 6.91177 | 6.92827 | 6.92315 | +0.01650 | +0.01138 | −0.00512 |
| OLMoE `nf4`, c4val1 | 18.90600 | 18.89285 | 19.06063 | −0.01315 | +0.15464 | **+0.16778** |

- **Every check passed:**
  - **The m arm engaged.** Its B=1 census ran `_gemm_nf4_grouped` 64 times per step on Granite and 32 on OLMoE (2 × layers),
    with no NF4 GEMV.
  - **Neither m nor t is bit-equal to g** on any text.
  - **All three arms of a text scored the same text** (`text_sha`).
  - **The premise held on the card:** both row-exact files 6/6, K25's contract compiled 30/30.
- **`BASELINE_SHIFT`** (reported, not gated): Granite yes (c4val1 m − g −0.078), OLMoE no.
- **g and t reproduce P93's OFF and ON exactly** on all four texts (5.36059 / 11.57477 / 6.91177 / 18.90600 and
  5.33507 / 11.59831 / 6.92315 / 19.06063). P93 ran on another host (Intel Xeon, driver 595.71.05); this one is an AMD
  Ryzen 9 7950X at driver 575.57.08.

## What the three arithmetics show

- **The comparison the `auto` default needs reads outside the gate in both families.**
  - Against the served M-tile, K25 moves c4val1 by +0.102 (Granite) and +0.168 (OLMoE).
  - On wikitext it stays inside: −0.020 and −0.005.
- **The served M-tile is not a quiet baseline.**
  - On Granite c4val1 it sits −0.078 from the GEMV. Production already serves T == 1 rows (GEMV) and B=16 rows (M-tile)
    with arithmetics whose K8 differ by more than the gate on that text.
  - On OLMoE the M-tile is within 0.017 of the GEMV on both texts.
- **P93's OLMoE swing is not the GEMV-to-tile difference** that production carries. The prediction that m − g would land
  near +0.10 to +0.20 on OLMoE c4val1 is refuted: −0.013. K25 is the arithmetic that moves OLMoE c4val1, by +0.155 from
  the GEMV and +0.168 from the M-tile.
- **On wikitext, every pair among the three arithmetics is within 0.026** in both families. On c4val1 the pairs range
  from 0.013 to 0.168.

**An interpretation, not a registered reading.**
- grouped-nf4-gemm's K27 read the TF32 tree's rms error against fp64 equal to the served kernel's, to four digits, at
  these families' B=16 shapes. So arithmetics of equal per-GEMM error move c4val1 K8 by up to ~0.17 ppl at 2048 tokens.
- Two explanations fit one draw per arithmetic:
  - c4val1's K8 has a realization spread of that size across equal-error arithmetics;
  - K25 has a specific bias on OLMoE. Both K25 precisions moved OLMoE c4val1 by more than 0.1 (P92 −0.107, P93 +0.155);
    the M-tile did not.
- This lane cannot separate them. A calibration lane can: several equal-error arithmetics per family, with the decision
  rule fixed before any data.

## Against the predictions

| prediction | outcome |
|---|---|
| \|t − m\| < 0.01 on both texts in both families | **refuted**: Granite 0.020 / 0.102, OLMoE 0.005 / 0.168 |
| OLMoE c4val1 m − g about +0.10 to +0.20 (BASELINE_SHIFT) | **refuted**: −0.013, no shift |
| Granite \|m − g\| ≈ 0.02–0.03 | **refuted**: 0.005 (wikitext), 0.078 (c4val1, BASELINE_SHIFT) |
| t − g reproduces P93's ON − OFF exactly | **held**: every g and t value equals P93's OFF and ON |

## Descriptive: the m arm at B=1 (not gated)

The served M-tile at T == 1 is an instrument arm, not a candidate route.
- Timed graph step at B=1: 6.93 ms (Granite), 5.77 ms (OLMoE).
- `_gemm_nf4_grouped`: 4.63 ms of Granite's censused step.

No ratio is quoted: the lane timed no GEMV arm on this host.

## The run

- **`p94-5090-1`:** NOT_RUN at the pre-flight (download bandwidth 21.8 MB/s, below 40), $0.0164, destroyed.
- **`p94-5090-2`:**
  - **Box:** one RTX 5090 (driver 575.57.08, power limit 600 W) on a Vast.ai verified host (machine 37958, AMD Ryzen 9
    7950X).
  - **Software:** e4b at `966a95e`, grouped-nf4-gemm at `908a2ca`, torch 2.8.0+cu128, triton 3.4.0, transformers
    5.16.1, bitsandbytes 0.50.1.
  - **Timeline (UTC):** launched 2026-10-01 23:48:30, lane started 23:49:44, lane complete 2026-10-02 00:05:30,
    destroyed 00:05:31 and proven absent.
  - **Cost:** $0.1373.
- **Lane total:** $0.1537, against a ceiling of $1.50.

## What follows

- **The default stays `0`.** `E4B_NF4_GROUPED_SMALLM=auto` remains an opt-in. P93 measured it on Granite `r12epi` at
  ×0.594 B=16, with K8 inside the gate against the GEMV.
- **The quality question is now the instrument's.** How far do equal-error arithmetics move K8 on c4val1, per family?
  The next lane registers that calibration before any noise-aware gate is applied, such as P44's KL from the bf16
  reference with its control.
- **The patch release ships the TF32 route as an opt-in** (#834), without a default change.

## Receipts

[`receipts/p94-5090-2/`](receipts/p94-5090-2/) holds:
- the twelve K8 JSONs;
- the m arm's two B=1 timed JSONs and its two censuses;
- verdict, summary, forensics, versions;
- the teardown proof and `SHA256SUMS`.
