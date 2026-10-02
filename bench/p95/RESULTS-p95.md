# P95 — results: **UNDER_RESOLVED on every text in both families**. Across arithmetics of equal per-GEMM error, K8's per-window spread is 0.054 (Granite c4val1), 0.055 (OLMoE c4val1), 0.026 and 0.030 (wikitext), so a single-window 0.05 gate cannot resolve these families. Window 0 reproduces P94 bit for bit on a third host

Registration: `bench/p95/PREREG-p95.md` (#855, `7b8bbbd`). Issue: #564. grouped-nf4-gemm is pinned at P94's `908a2ca`.

**Verdict by `p95_reduce.py`: `UNDER_RESOLVED`.** σ = max(SD(m − g), SD(t − m)) over the fresh windows:

| family, text | fresh windows | SD(m − g) | SD(t − m) | **σ** | windows a gate needs (W = ⌈(σ/0.025)²⌉) |
|---|---:|---:|---:|---:|---:|
| Granite `r12epi`, c4val1 | 8 | 0.0382 | 0.0535 | **0.0535** | 5 |
| Granite `r12epi`, wikitext | 4 | 0.0258 | 0.0250 | **0.0258** | 2 |
| OLMoE `nf4`, c4val1 | 8 | 0.0546 | 0.0525 | **0.0546** | 5 |
| OLMoE `nf4`, wikitext | 4 | 0.0120 | 0.0303 | **0.0303** | 2 |

Under the registered consequence:
- **P94's verdict stands.** This lane licenses nothing.
- **Any later quality lane on these families**, including one that asks again about K25's default, registers a
  windowed K8 gate:
  - |mean delta over W fresh windows| ≤ 0.05 on every text;
  - W ≥ 5 on c4val1 and W ≥ 2 on wikitext;
  - windows disjoint from P94's and P95's (k ≥ 9);
  - its rule fixed before its data.

**Every check passed:**
- the m arm engaged (the B=1 census: `_gemm_nf4_grouped` 64 / 32 per step, no NF4 GEMV);
- all 84 arms ran;
- every window's three arms scored one text, and the 28 window texts are distinct;
- no arm is bit-equal to g in every fresh window.

## Every window (K8 ppl)

| window | Granite c4val1 g / m / t | m − g | t − m | OLMoE c4val1 g / m / t | m − g | t − m |
|---|---|---:|---:|---|---:|---:|
| 0 (P94's) | 11.57477 / 11.49663 / 11.59831 | −0.078 | +0.102 | 18.90600 / 18.89285 / 19.06063 | −0.013 | +0.168 |
| 1 | 12.68226 / 12.64567 / 12.58738 | −0.037 | −0.058 | 18.03455 / 18.11928 / 18.17727 | +0.085 | +0.058 |
| 2 | 8.15270 / 8.18000 / 8.11296 | +0.027 | −0.067 | 12.71239 / 12.63151 / 12.72517 | −0.081 | +0.094 |
| 3 | 13.64093 / 13.70230 / 13.69803 | +0.061 | −0.004 | 14.02333 / 13.99369 / 13.95035 | −0.030 | −0.043 |
| 4 | 8.80934 / 8.84909 / 8.82301 | +0.040 | −0.026 | 14.48223 / 14.44858 / 14.54336 | −0.034 | +0.095 |
| 5 | 8.81989 / 8.86437 / 8.77369 | +0.044 | −0.091 | 14.88273 / 14.90535 / 14.93471 | +0.023 | +0.029 |
| 6 | 11.49006 / 11.49062 / 11.55159 | +0.001 | +0.061 | 9.81906 / 9.79594 / 9.82467 | −0.023 | +0.029 |
| 7 | 8.12665 / 8.08213 / 8.12915 | −0.045 | +0.047 | 13.37436 / 13.42165 / 13.39277 | +0.047 | −0.029 |
| 8 | 10.41431 / 10.43467 / 10.41920 | +0.020 | −0.016 | 15.99265 / 15.94464 / 15.93680 | −0.048 | −0.008 |

| window | Granite wikitext g / m / t | m − g | t − m | OLMoE wikitext g / m / t | m − g | t − m |
|---|---|---:|---:|---|---:|---:|
| 0 (P94's) | 5.36059 / 5.35515 / 5.33507 | −0.005 | −0.020 | 6.91177 / 6.92827 / 6.92315 | +0.017 | −0.005 |
| 1 | 9.90992 / 9.96538 / 9.99425 | +0.055 | +0.029 | 11.08387 / 11.10082 / 11.08843 | +0.017 | −0.012 |
| 2 | 4.86012 / 4.88216 / 4.87235 | +0.022 | −0.010 | 9.32107 / 9.32961 / 9.34263 | +0.009 | +0.013 |
| 3 | 4.56896 / 4.57127 / 4.58794 | +0.002 | +0.017 | 8.38879 / 8.38045 / 8.39124 | −0.008 | +0.011 |
| 4 | 6.68552 / 6.68496 / 6.65882 | −0.001 | −0.026 | 11.04759 / 11.06504 / 11.01285 | +0.018 | −0.052 |

## What it shows

- **A single window cannot resolve 0.05 on c4val1.** Production's own two arithmetics, the GEMV at T == 1 and the
  M-tile above it, differ by more than 0.05 on 1 of the 8 fresh Granite windows and 2 of the 8 OLMoE ones (up to
  0.085). K25 against the M-tile exceeds 0.05 on 4 of 8 and 3 of 8. Every pair here has equal per-GEMM rms error
  (grouped-nf4-gemm K27).
- **Over the fresh windows, K25 shows no consistent shift** against the M-tile. The registered test is
  |mean| > t₀.₉₇₅(7) · SE:
  - Granite c4val1: −0.019 (SE 0.019);
  - OLMoE c4val1: +0.028 (SE 0.019);
  - wikitext: +0.002 and −0.010.
- **P94's window was an outlier for K25.** P94's t − m on OLMoE c4val1 (+0.168) sits about 2.7 SD above the fresh
  windows' mean, and Granite's (+0.102) about 2.3 SD. P92, P93 and P94 all scored that one window.
- **These means are descriptive.** P95 registered that it licenses nothing. Asking about K25's default again takes a
  new lane, on new windows, with the windowed gate fixed in advance.

## Against the predictions

| prediction | outcome |
|---|---|
| window 0 reproduces P94 exactly | **held** on all twelve values, on a third host |
| c4val1 σ 0.04–0.10 (UNDER_RESOLVED), W 3–16 | **held**: 0.054 / 0.055, W = 5 |
| wikitext σ ≤ 0.025 | **refuted**: 0.026 / 0.030 (W = 2) |
| Granite c4val1 m − g mean near 0 | **held**: +0.014 (SE 0.014) |
| OLMoE c4val1 t − m: no consistent shift | **held**: +0.028 (SE 0.019) |

## The run

- **Proving rentals:**
  - `p95-prove-1`: REFUSED at $0. A machine exclusion named another lane's single bandwidth refusal.
  - `p95-prove-2`: NOT_RUN at the bandwidth pre-flight, 25.0 MB/s, $0.0135.
  - `p95-prove-3`: NOT_RUN at the bandwidth pre-flight, 2.3 MB/s, $0.0359.
  - `p95-prove-4`: OK, `PROVED` on sm_120, $0.0422.
- **`p95-5090-1`:**
  - **Box:** one RTX 5090 (driver 580.178.04, power limit 400 W) on a Vast.ai verified host (machine 29048, AMD
    Ryzen Threadripper PRO 7975WX).
  - **Software:** e4b at `7b8bbbd`, grouped-nf4-gemm at `908a2ca`, torch 2.8.0+cu128, triton 3.4.0, transformers
    5.16.1.
  - **Timeline (UTC):** launched 03:21:52, lane started 03:23:04, lane complete 04:34:35, destroyed 04:34:36 and
    proven absent.
  - **Cost:** $0.7049.
- **Lane total:** $0.7965, against a ceiling of $3.00.
- **Window 0 on three hosts.** P93 (Intel Xeon, driver 595.71.05), P94 (AMD Ryzen 9 7950X, 575.57.08) and this
  run (AMD Threadripper PRO 7975WX, 580.178.04) read the same K8 to five decimals, where they share arms.

## What follows

- **The gate for these families is the windowed one.** The rule is above.
- **The next lane asks K25's default again** on fresh windows (k ≥ 9) with that gate, registered before its data.
  P93's speed (B=16 ×0.594 / ×0.598) is already read.

## Receipts

[`receipts/p95-5090-1/`](receipts/p95-5090-1/) holds:
- the 84 K8 JSONs (`<family>_k8_<arm>_<text>_w<k>.json`);
- the m arm's two B=1 timed JSONs and its two censuses;
- verdict, summary, forensics, versions;
- the teardown proof and `SHA256SUMS`.
