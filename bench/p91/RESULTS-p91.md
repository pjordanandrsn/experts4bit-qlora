# P91 — results: **READ**. The NF4 expert GEMM is 62 % (Granite) and 72 % (OLMoE) of the NF4 families' B=16 decode kernel time on an RTX 5090, so the next family lane is an NF4 grouped small-M kernel

Registration: `bench/p91/PREREG-p91.md` (#825, `facf8da`). Issue: #564. This lane is descriptive: no change is tested
and nothing is licensed.

**Verdict by `p91_reduce.py`: READ.** The registered decision names `nf4-grouped-small-m-kernel`.

```
P91_VERDICT READ next_lane=nf4-grouped-small-m-kernel | NF4 grouped share at B=16: granite 61.7%, olmoe 71.9% (>= 40% in at least one family)
```

| family (position config) | B=16 step | census | NF4 expert kernel | B=1 step | census | NF4 decode GEMV |
|---|---:|---:|---:|---:|---:|---:|
| Granite-3.1-3B-A800M (`r12epi`) | 9.192 ms | 8.872 (−3.5 %) | **5.479 ms, 61.7 %** | 3.338 ms | 3.328 (−0.3 %) | 1.688 ms, 50.7 % |
| OLMoE-1B-7B (`nf4`) | 12.471 ms | 11.975 (−4.0 %) | **8.607 ms, 71.9 %** | 3.956 ms | 3.612 (−8.7 %) | 1.651 ms, 45.7 % |

The B=16 NF4 kernel is `_gemm_nf4_grouped`, at 64 calls per step on Granite (32 layers × gate_up + down) and 32 on
OLMoE (16 layers). The B=1 kernel is `_gemv_nf4_grouped`, at the same counts. Every census reconciles within the registered 10 %.

## The run

- **`p91-5090-1`:** one RTX 5090 (driver 580.119.02, power.limit 475 W) on an AMD EPYC 9655 (machine 150710, also
  P89's reading box). e4b 0.37.8 at `facf8da`, grouped-nf4-gemm at `4cc831c`, torch 2.8.0, triton 3.4.0, transformers
  5.16.1.
- **Timeline (UTC).** Rented 15:43:18; destroyed 15:51:47, absent.
- **Cost:** $0.0801 of the $1.50 ceiling. No proving rental (1.0 h guard).
- **The A2000 rehearsal** ran the whole path through fetch and bake. Its census arms stopped at the fp8 paged-KV append,
  which sm_86 cannot compile.

## Where the rest goes

- **Granite B=16,** besides the NF4 GEMM:
  - cutlass bf16 GEMMs (the attention projections), 1.09 ms (12.3 %);
  - fp8 paged decode, 0.57 (6.4 %);
  - an `indexSelect`, 0.39 (4.4 %);
  - a `gemmSN_TN` (a skinny cuBLAS GEMM), 0.27 (3.1 %).
- **OLMoE B=16:** cutlass bf16 GEMMs 0.94 ms (7.8 %), fp8 paged decode 0.55 (4.6 %), then elementwise kernels.
- **At B=1 on both families,** a cuBLAS GEMV (`gemvx`) is the second item: 0.71 ms on Granite, 0.63 on OLMoE. These are
  the attention projections at one row.

## Against the predictions

| prediction | outcome |
|---|---|
| Granite B=16 NF4 grouped 40–60 % | **61.7 %**, just above the band |
| OLMoE B=16 NF4 grouped 50–70 % | **71.9 %**, just above the band |
| B=1 NF4 GEMVs 30–50 % | 50.7 % (Granite, at the edge) and 45.7 % (OLMoE) |
| every census within 5 % | three within 5 %; OLMoE B=1 at −8.7 %, inside the registered 10 % but outside my prediction |
| the decision names the NF4 small-M kernel | **held** |

I underestimated the NF4 kernel's share on both families by a few points.

## What follows (the registered decision)

An **NF4 grouped small-M kernel**: K19's skeleton (16-row expert tiles, the in-kernel gather, sorted output, bf16
MMA) with NF4 dequant (the 16-entry codebook, the per-64 absmax), for the NF4 stores' batched rows.

grouped-nf4-gemm's K22/K24 benches put today's NF4 grouped GEMM at about 25 % of the byte floor on gpt-oss's shapes,
where K21 reached 50 %. The new kernel changes arithmetic, from TF32 MMA on fp32 dequantized weights to bf16. So the
lane carries a per-family quality gate: K8 on Granite, whose licence is K8, and on OLMoE against its NF4 position,
before any default.

## Receipts

[`receipts/p91-5090-1/`](receipts/p91-5090-1/) holds the four timed JSONs, the four censuses, verdict, summary,
forensics, versions, the teardown proof, and `SHA256SUMS`.
