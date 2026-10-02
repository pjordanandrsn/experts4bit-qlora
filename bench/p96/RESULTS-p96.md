# P96 — results: **LICENSED**. Under P95's windowed K8 gate, K25 against the served NF4 M-tile reads mean t − m −0.001 / −0.015 (Granite c4val1 / wikitext) and −0.016 / −0.004 (OLMoE), inside 0.05 on every text, so `E4B_NF4_GROUPED_SMALLM` defaults to `auto`

Registration: `bench/p96/PREREG-p96.md` (#870, `c46f72a`). Issue: #564. grouped-nf4-gemm is pinned at `908a2ca`.

**Verdict by `p96_reduce.py`: `LICENSED`.** |mean(t − m)| ≤ 0.05 over the fresh windows on every text in both
families. Under the registered consequence, with P93's speed (B=16 ×0.594 / ×0.598), `E4B_NF4_GROUPED_SMALLM` defaults
to `auto`. That covers rows above T == 1 only; T == 1 stays on the GEMV.

| family, text | windows | mean(t − m) | SD | SE | P95's σ |
|---|---:|---:|---:|---:|---:|
| Granite `r12epi`, c4val1 | 8 (9–16) | **−0.0009** | 0.0396 | 0.0140 | 0.0535 |
| Granite `r12epi`, wikitext | 4 (9–12) | **−0.0146** | 0.0402 | 0.0201 | 0.0258 |
| OLMoE `nf4`, c4val1 | 8 (9–16) | **−0.0158** | 0.0770 | 0.0272 | 0.0546 |
| OLMoE `nf4`, wikitext | 4 (9–12) | **−0.0040** | 0.0130 | 0.0065 | 0.0303 |

**Every check passed:**
- **Engagement:** the B=1 censuses ran `_gemm_nf4_grouped` (m) and `_gemm_nf4_grouped_smallm` (t) 64 / 32 times per
  step (2 × layers), with neither arm running the other's kernel or the NF4 GEMV.
- **Completeness:** all 48 arms ran, with none skipped at the deadline.
- **Text integrity:** every window's two arms scored one text, and the windows' texts are distinct.
- **Engaged:** t is not bit-equal to m in any window.

## Every window (K8 ppl, t − m)

| window | Granite c4val1 m / t | t − m | OLMoE c4val1 m / t | t − m |
|---|---|---:|---|---:|
| 9 | 10.75860 / 10.72491 | −0.034 | 19.80839 / 19.82023 | +0.012 |
| 10 | 8.29315 / 8.29028 | −0.003 | 19.23804 / 19.20304 | −0.035 |
| 11 | 11.98638 / 12.06938 | +0.083 | 13.43343 / 13.52214 | +0.089 |
| 12 | 11.60022 / 11.60583 | +0.006 | 14.01362 / 13.92417 | −0.089 |
| 13 | 10.95330 / 10.89975 | −0.054 | 10.40589 / 10.44356 | +0.038 |
| 14 | 10.86637 / 10.87056 | +0.004 | 6.79257 / 6.76383 | −0.029 |
| 15 | 10.13913 / 10.13320 | −0.006 | 11.34011 / 11.37868 | +0.039 |
| 16 | 4.30076 / 4.29701 | −0.004 | 29.53053 / 29.38061 | −0.150 |

| window | Granite wikitext m / t | t − m | OLMoE wikitext m / t | t − m |
|---|---|---:|---|---:|
| 9 | 9.91566 / 9.84140 | −0.074 | 7.33671 / 7.34599 | +0.009 |
| 10 | 7.73037 / 7.73908 | +0.009 | 7.73129 / 7.72092 | −0.010 |
| 11 | 4.98764 / 4.98503 | −0.003 | 8.50468 / 8.48568 | −0.019 |
| 12 | 7.30532 / 7.31511 | +0.010 | 8.92635 / 8.93055 | +0.004 |

## What it shows

- **The windowed means sit near zero.** Single windows still swing by up to 0.15. Six of the 24 exceed 0.05: three
  on OLMoE c4val1, two on Granite c4val1, one on Granite wikitext, in both directions. A single-window gate would have
  failed this change as it failed P94's. The windowed gate reads it as no shift.
- **Window size scales the swing.** OLMoE c4val1's largest delta (−0.150) is on window 16, whose perplexity is 29.5;
  in relative terms that is −0.5 %. A K8 delta in ppl grows with the window's ppl, so a later instrument might gate
  on the mean nll difference. That is noted, not applied.
- **The 95 % intervals on the means:**
  - Granite c4val1 −0.001 ± 0.033 and wikitext −0.015 ± 0.064;
  - OLMoE c4val1 −0.016 ± 0.064 and wikitext −0.004 ± 0.021.
  - The gate is on the mean, as registered. The intervals say how much it can resolve at these window counts.
- **Descriptive, not gated:** B=1 graph step with the K25 arm is 2.80 ms (Granite) and 3.11 ms (OLMoE), against the
  M-tile's 6.66 / 5.46 at T == 1. The default leaves T == 1 on the GEMV, which this lane did not time.

## Against the predictions

| prediction | outcome |
|---|---|
| LICENSED, \|mean t − m\| < 0.04 on every text | **held**: the largest is 0.016 |
| per-window SD within ×1.5 of P95's σ | **partly refuted**: Granite c4val1 ×0.74, OLMoE c4val1 ×1.41 inside; Granite wikitext ×1.56 and OLMoE wikitext ×0.43 outside |
| engagement 64 / 32 calls per step for each arm's kernel | **held** |

## The run

- **`p96-prove-1`:**
  - The first attempt was REFUSED at $0: all three account rental slots were held by lane TC's runs.
  - The relaunch went through a slot watcher: OK, `PROVED` on sm_120, $0.0626.
- **`p96-5090-1`:**
  - **Box:** one RTX 5090 (driver 595.71.05, power limit 500 W) on a Vast.ai verified host (machine 45501, AMD Ryzen 9
    3900X).
  - **Software:** e4b at `c46f72a`, grouped-nf4-gemm at `908a2ca`, torch 2.8.0+cu128.
  - **Timeline (UTC):** launched 05:30:54, lane started 05:31:50, lane complete 06:39:04, destroyed 06:39:05 and
    proven absent.
  - **Cost:** $0.5799.
- **Lane total:** $0.6425, against a ceiling of $2.50.

## What follows

- **`E4B_NF4_GROUPED_SMALLM` defaults to `auto`**, in a separate PR (the registered consequence). The NF4 store's
  batched decode rows (T > 1) take K25 at the served precision when the installed grouped-nf4-gemm carries it.
  - T == 1 keeps the GEMV.
  - `0` restores the previous route.
  - An older kernel package keeps the previous route silently.
- The NF4 families' B=16 decode is then about 40 % faster by default: P93's ×0.594 (Granite), ×0.598 (OLMoE).

## Receipts

[`receipts/p96-5090-1/`](receipts/p96-5090-1/) holds:
- the 48 K8 JSONs (`<family>_k8_<arm>_<text>_w<k>.json`);
- the four B=1 timed JSONs and their censuses;
- verdict, summary, forensics, versions;
- the teardown proof and `SHA256SUMS`.
