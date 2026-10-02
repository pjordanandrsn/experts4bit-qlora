# P92 — results: **Granite LICENSED (×0.937), OLMoE QUALITY_FAIL (and ×1.024 slower)**. K25 runs at about the served NF4 GEMM's speed; Granite's gain is the glue it folds away

Registration: `bench/p92/PREREG-p92.md` (#828, `a626ae3`). Issue: #564. The kernel is grouped-nf4-gemm K25 (#429,
`8cc3510`) through `E4B_NF4_GROUPED_SMALLM` (#827).

**Verdict by `p92_reduce.py`: `granite=LICENSED olmoe=QUALITY_FAIL`.** Under the registered consequence the default
stays `0`: it moves to `auto` only if both families read LICENSED.

| family (config) | B=16 OFF → ON (mean of 2 draws) | B=1 OFF → ON | K8 wikitext OFF → ON | K8 c4val1 OFF → ON | verdict |
|---|---:|---:|---:|---:|---|
| Granite-3.1-3B-A800M (`r12epi`) | 10.529 → 9.865 ms, **×0.937** | 3.860 → 3.754, ×0.972 | 5.36059 → 5.33725 (−0.023) | 11.57477 → 11.59106 (+0.016) | **LICENSED** |
| OLMoE-1B-7B (`nf4`) | 14.355 → 14.694 ms, ×1.024 | 4.232 → 4.512, ×1.066 | 6.91177 → 6.92890 (+0.017) | 18.90600 → 18.79889 (**−0.107**) | **QUALITY_FAIL** |

- **Every engagement check held.** ON ran `_gemm_nf4_grouped_smallm` exactly 64 times (Granite) and 32 times (OLMoE)
  per step at both B=16 and B=1, with no served NF4 kernel. OFF ran none of it.
- **The draws agree.** B=16 draws were at most 0.18 % apart; K8 ON differs from OFF on every text.
- **OLMoE's quality failure:** its c4val1 K8 moved −0.107 ppl. That is an improvement, but the rule for an uncalibrated
  change is two-sided (|Δ| ≤ 0.05): a large move in either direction is a failure.

## Where the time goes (census, OFF vs ON, ms per step)

| | served NF4 GEMM (OFF) | K25 (ON) | the rest, ON − OFF |
|---|---:|---:|---:|
| Granite B=16 | 6.602 | 6.464 (−2 %) | −0.35 (the `[R, H]` gather and the unsort folded away) |
| OLMoE B=16 | 10.388 | 10.852 (+4 %) | −0.17 |
| Granite B=1 | 1.980 (`_gemv_nf4_grouped`) | 2.025 | −0.12 |
| OLMoE B=1 | 1.983 (`_gemv_nf4_grouped`) | 2.336 | −0.06 |

- **K25 is not faster than the served GEMM at these shapes.** Granite's ×0.937 is almost all glue (the expansion,
  index and scatter kernels the lean route drops); its GEMM is 2 % faster. On OLMoE, K25 is 4 % slower, so the step is
  too.
- **The NF4 decode is the likely shared bottleneck.** Both kernels decode each nibble through a codebook lookup. K25
  loads one int64 per byte; the served kernel gathers from a register table. K19, which decodes int4 with shifts and no
  loads, reached 89 % of its byte floor (K20). This is inferred from the two kernels' shared decode and not measured
  here: no profiler counter was read.

## Against the predictions

| prediction | outcome |
|---|---|
| B=16 ON/OFF Granite 0.66–0.85 | **0.937**, outside: the kernel is not 1.5–2× the served one |
| B=16 ON/OFF OLMoE 0.65–0.82 | **1.024**, outside, slower |
| B=1 ON/OFF 0.95–1.20 | 0.972 and 1.066, held |
| \|ΔK8\| < 0.01 ppl | refuted in both families: 0.016–0.023 on Granite, 0.017 and **0.107** on OLMoE |
| engagement holds | held |
| OFF reproduces P91's steps | no: 10.53 / 14.36 ms here against 9.19 / 12.47, +15 % (below) |

I assumed K25 would match K21's efficiency on the MXFP4 store. K21 decodes with integer arithmetic, so the assumption
did not transfer to a codebook format.

## The run

- **`p92-5090-2`:** one RTX 5090 (driver 595.71.05, power.limit 575 W) on an Intel Xeon E5-2698 v4. e4b 0.37.8 at
  `a626ae3`, grouped-nf4-gemm at `8cc3510`, torch 2.8.0, triton 3.4.0, transformers 5.16.1.
- **Premise on the card:** the K25 row-exact test passed, and K25's contract compiled on sm_120, 28/28.
- **Timeline (UTC):** rented 17:22:52; destroyed 17:57:43, absent.
- **`p92-5090-1`** was NOT_RUN: its box's ssh refused connections for 180 s. It was destroyed at 17:21:56 and proven
  absent, for $0.0334.
- **Cost:** $0.3600 for the lane ($0.0334 + $0.3266), against the $1.50 ceiling.
- **Host spread, reported, not gated.** This box read OFF 15 % slower than P91's (AMD EPYC 9655, 475 W, driver
  580.119.02): Granite 10.53 vs 9.19 ms, OLMoE 14.36 vs 12.47 ms. The served GEMM's own GPU time differed too (Granite
  6.60 vs 5.48 ms), so the gap is not host-only. Every ratio here is within one box.

## What follows (the registered consequence)

- **The default stays `0`.** `E4B_NF4_GROUPED_SMALLM=auto` is measured on Granite's `r12epi` at ×0.937 with its K8
  inside the gate. That is an opt-in for that family, not a default.
- **For OLMoE the registered pointer is the weight rounding.** A K25 whose weight operand keeps the served kernel's
  precision (fp32 weights, TF32 MMA) would remove the arithmetic change. On this read it would also be slower.
- **The speed lever for both families is the NF4 decode,** not the MMA or the glue. That is the next kernel question,
  for a lane that profiles before it builds.

## Receipts

[`receipts/p92-5090-2/`](receipts/p92-5090-2/) holds:
- the twelve timed JSONs and the eight censuses;
- the eight K8 JSONs;
- verdict, summary, forensics, versions;
- the teardown proof and `SHA256SUMS`.
