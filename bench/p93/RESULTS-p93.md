# P93 — results: **Granite LICENSED (×0.594), OLMoE QUALITY_FAIL (c4val1 K8 +0.155 ppl)**. The served-precision route is about 40 % faster at B=16 in both families. OLMoE's c4val1 K8 moved the other way from P92's, so its failure is not the weight rounding

Registration: `bench/p93/PREREG-p93.md` (#836, `2c7fec2`). Issue: #564. The route is #834 (K25-tree through TF32 MMA at
K27's plan). grouped-nf4-gemm is pinned at `908a2ca`.

**Verdict by `p93_reduce.py`: `granite=LICENSED olmoe=QUALITY_FAIL`.** Under the registered consequence the default
stays `0`: it moves only if both families read LICENSED.

| family (config) | B=16 OFF → ON (mean of 2 draws) | B=1 OFF → ON | K8 wikitext OFF → ON | K8 c4val1 OFF → ON | verdict |
|---|---:|---:|---:|---:|---|
| Granite (`r12epi`) | 10.535 → 6.258 ms, **×0.594** | 3.862 → 3.299, ×0.854 | 5.36059 → 5.33507 (−0.026) | 11.57477 → 11.59831 (+0.024) | **LICENSED** |
| OLMoE (`nf4`) | 14.352 → 8.585 ms, ×0.598 | 4.231 → 3.636, ×0.859 | 6.91177 → 6.92315 (+0.011) | 18.90600 → 19.06063 (**+0.155**) | **QUALITY_FAIL** |

- **Every check passed.** Engagement held in every census (64 / 32 K25 calls per step ON, no served NF4 kernel; none OFF).
  B=16 draws were at most 0.18 % apart. K8 ON differs from OFF on every text. The premise held on the card: the row-exact
  test 4/4, K25's contract compiled 30/30.
- **The OFF K8 values reproduce P92's exactly** on every text (5.36059, 11.57477, 6.91177, 18.90600).

## Where the time goes (census, ms per step)

| | served NF4 kernel (OFF) | K25 TF32 tree (ON) | step OFF → ON |
|---|---:|---:|---:|
| Granite B=16 | 6.606 (`_gemm_nf4_grouped`) | 2.700 (0.41×) | 10.566 → 6.302 |
| OLMoE B=16 | 10.389 | 4.954 (0.48×) | 14.252 → 8.630 |
| Granite B=1 | 1.982 (`_gemv_nf4_grouped`) | 1.564 | 3.869 → 3.328 |
| OLMoE B=1 | 1.982 | 1.433 | 4.252 → 3.643 |

In-model, the expert GEMM runs at 0.41× / 0.48× of the served kernel, close to K27's bench (0.448 / 0.502). The lean glue
contributes the rest. At B=1, the tree beats the scalar decode GEMV.

## OLMoE's K8: a failure that is not the weight rounding

P92 read OLMoE's c4val1 K8 −0.107 ppl with K25 in bf16, and attributed it to bf16 weight rounding. P93 changes exactly
that rounding: the same route, at the served kernel's precision. Two things refute the attribution:
- **OLMoE's c4val1 now moves +0.155**, the other sign and further.
- **Granite moves about as much at TF32 as at bf16** (−0.026 / +0.024 against −0.023 / +0.016). TF32 rounds the weight
  8× less than bf16 (2^-11 against 2^-8 relative).

What the two runs share, against OFF at T = 1 (which K8 reads), is the route: the device-grouped tile path and the K25
kernel's accumulation, against the scalar fp32 GEMV. The arithmetic the weights go through is not what they share. So
the OLMoE c4val1 K8 moves by ±0.1–0.15 ppl when the route's arithmetic changes, in either direction. Whether the
served M-tile path, today's B=16 production arithmetic, moves it as much against the GEMV is unmeasured. It is the
question the next lane should register.

That question is a calibration of the instrument; it is not a new gate. If the production path's own arithmetic
variants move OLMoE c4val1 by ~0.1, then a 0.05-ppl gate on that text is below the instrument's spread for this family.
A noise-aware instrument would then be needed, such as P44's KL from the bf16 reference with its control, registered
before it is applied. Until then the rule stands, and OLMoE's default stays where it is.

## Against the predictions

| prediction | outcome |
|---|---|
| B=16 ×0.55–0.75 in both | **held**: 0.594 / 0.598 |
| B=1 ×0.70–0.95 | **held**: 0.854 / 0.859 |
| \|ΔK8\| < 0.02 | **refuted**: Granite 0.026 / 0.024; OLMoE 0.011 / **0.155** |
| both LICENSED, the default moves | refuted: OLMoE QUALITY_FAIL |

## The run

- **`p93-5090-1`:** one RTX 5090 (driver 595.71.05, power.limit 575 W; the same card as P92's) on an Intel Xeon host.
  e4b at `2c7fec2`, grouped-nf4-gemm at `908a2ca`, torch 2.8.0, triton 3.4.0.
- **Timeline (UTC):** launched 21:17:03, lane complete 21:52:15, destroyed and proven absent. **Cost:** $0.3317.

## What follows

- **The default stays `0`.** `E4B_NF4_GROUPED_SMALLM=auto` is now measured on Granite `r12epi` at ×0.594 B=16 and ×0.854
  B=1, with K8 inside the gate: a strong opt-in for that family.
- **OLMoE's quality question moves to the instrument.** How much do the production path's own arithmetic variants move
  K8 on OLMoE c4val1? That is the next lane's question, pre-registered with its decision rule.

## Receipts

[`receipts/p93-5090-1/`](receipts/p93-5090-1/) holds:
- the twelve timed JSONs and the eight censuses;
- the eight K8 JSONs;
- verdict, summary, forensics, versions;
- the teardown proof and `SHA256SUMS`.
