# P42 — where the B=16 decode step goes

Pre-registration: [`P42-PREREG.md`](P42-PREREG.md), including Amendment 1. Receipts:
`receipts/experts4bit-qlora/2026-09-11/p42-census-5` in the private record. Every number
here is read from a census or a timed receipt that box wrote.

**The headline: at B=16 the int4 attention store buys nothing.** It costs what plain
bf16 costs, because at M > 1 it dequantises and runs the same cuBLAS/CUTLASS GEMM. Its
saving is real and large at B=1, and it disappears the moment you batch.

## Gates first

| registered | measured | |
|---|---|---|
| STOP-1: `int4_b16` within ±5 % of p39-box4's calibrated 12.390 ms | **12.849 ms**, 3.7 % | PASS |
| Amendment 1 cross-check: uncalibrated attention swap reports 192 projections | **192** | PASS |
| STOP-3: all four censuses written | 4 of 4 | PASS |

So the census describes the configuration we quote, and the uncalibrated shortcut did
not change it: `int4_b1` reads 4.853 ms where P39 measured 4.820 ms with a calibrated
pack, 0.7 % apart.

## The 2×2

| arm | step | throughput | census accounted |
|---|---|---|---|
| `nf4_b1` | 10.014 ms | | 89.3 % |
| `nf4_b16` | 33.557 ms | 476.8 tok/s | 95.6 % |
| `int4_b1` | 4.853 ms | | 88.2 % |
| `int4_b16` | **12.849 ms** | 1245.2 tok/s | 95.8 % |

The int4 path is worth 2.06× at B=1 and 2.61× at B=16 over NF4. That is not in question;
what follows is where the remaining 12.849 ms sits.

## `int4_b16`, by kernel

Self-CUDA over the profiled replays, so the unit is milliseconds of one decode step.

| kernel | ms/step | % of step | calls/step | ×(its B=1 cost) |
|---|---|---|---|---|
| `_gemv_int4_b32` | **6.662** | 51.9 | 96 | 4.0 |
| `cutlass_80_wmma_tensorop_bf16_*` | **2.227** | 17.3 | 240 | absent at B=1 |
| `_fp8_paged_decode_split_f8dot` | 0.678 | 5.3 | 48 | 2.0 |
| `indexSelectSmallIndex` | 0.465 | 3.6 | 48 | 9.3 |
| `cutlass_80_wmma_tensorop_bf16_*` (lm_head) | 0.377 | 2.9 | 1 | absent at B=1 |
| `_quant_x_rows` | 0.316 | 2.5 | 96 | 1.4 |
| `indexFuncSmallIndex` | 0.206 | 1.6 | 48 | absent at B=1 |
| `_reduce_partials` | 0.098 | 0.8 | 96 | 0.2 |

The expert GEMV takes **half the step** and scales at **4.0×** for 16× the tokens — it is
the largest row and the best-behaved one. Attention itself (`_fp8_paged_decode_split_f8dot`)
is 5.3 %, and scales 2.0×.

## The finding

Grouping by kernel family across all four arms:

| family | nf4_b1 | int4_b1 | nf4_b16 | int4_b16 |
|---|---|---|---|---|
| cuBLAS `gemvx` (M = 1 dense GEMV) | 1.954 / 241 calls | **0.494 / 49** | 0 | 0 |
| CUTLASS bf16 tensor-op (M > 1 GEMM) | 0 | 0 | 2.577 / 241 | **2.604 / 241** |

Read it across:

- **At B=1**, turning the attention projections to int4 drops the dense-GEMV family from
  1.954 ms to 0.494 ms. The work moves into the int4 GEMV. A **1.46 ms/step** saving.
- **At B=16**, the two arms are 2.577 and 2.604 ms over the same 241 calls. The int4 arm
  is **0.027 ms worse**. Nothing is saved.

This is `int4_attn.py` doing exactly what its docstring says: *"decode (M = 1) served by
the grouped int4 GEMV, prefill (M > 1) by dequant-then-matmul"*. At B=16 every decode
step has M = 16, so every attention projection takes the M > 1 branch, unpacks its int4
weights to bf16 and hands them to the same GEMM the unquantised path uses. The store
saves memory; at batch it saves no time.

The 241 calls are 192 attention projections + 48 routers + 1 lm_head, and the profiler
groups them under one kernel name, so the census **cannot** split the 2.604 ms between
them by name alone. The 192/48/1 split is the upper bound on how much of it the
attention projections could give back.

## The registered predictions, scored

1. **The two carried levers.** Registered band 1.0–2.0 ms each.
   - *small-M int4 attention GEMM*, predicted ≈1.7 ms: the family measures **2.227 ms**
     (2.604 including the lm_head row). Outside the band, so **refuted as stated** —
     understated by about a third, and in the favourable direction.
   - *fused permute/unpermute*, predicted ≈1.5 ms: the index/gather/scatter family
     (`indexSelectSmallIndex` + `indexFuncSmallIndex` + `vectorized_gather` +
     `_combine_rows`) measures **0.950 ms**. Outside the band, **refuted** — overstated by
     more than half, and it is third on the list, not first.

   Neither number was right. I said in the prereg I expected at least one to be wrong;
   both were, in opposite directions.

2. **The expert GEMV dominates.** `_gemv_int4_b32` at 51.9 %, the largest row by 3×.
   **Confirmed.**

3. **Batch scaling is superlinear somewhere.** Registered as "at least one kernel family
   costs more than 16× its B=1 cost". The worst measurable scaler is
   `indexSelectSmallIndex` at **9.3×**. **Refuted** — and the phrasing was poor: three
   families do not exist at B=1 at all, which is a stronger statement than any ratio and
   one the prediction had no way to express.

## The decision rule, applied

Registered before the measurement: *pursue the largest attributable cost that is not the
expert GEMV itself, whichever it turns out to be*. That is the **CUTLASS bf16 tensor-op
GEMM family, 2.227 ms/step, 17.3 %** — and the cross-arm comparison says the int4
attention projections inside it are paying full bf16 price.

A small-M int4 GEMM that keeps the weights packed would take that row toward what the
B=1 path already demonstrates is possible. Bounding it honestly:

| | ms/step | vs vLLM `graph_r1` 7.882 |
|---|---|---|
| today | 12.849 | 1.63× |
| minus the whole CUTLASS family (unreachable: includes router + lm_head) | 10.245 | 1.30× |
| minus the 192 attention projections' share, if it went to zero | ~10.8 | ~1.37× |

So this lever alone does not close the gap, and nothing here says it would. It is the
largest separable non-GEMV item, it is the one the decision rule names, and its size is
now measured rather than remembered.

## What this lane did not do

- It changed no kernel and measured no fix. Whether a small-M int4 GEMM beats
  dequant-then-GEMM at M=16 on this shape is unmeasured, and the crossover this
  repository already documents is the reason the current code chooses dequant there.
  That choice may be right; this census only shows what it costs.
- It cannot split the 241-call GEMM row between attention, router and lm_head. Doing so
  needs `record_function` regions around the three, which is a harness change.
- The expert GEMV's 6.662 ms is untouched by any of this. It is half the step, and the
  A2000 split-K work (gnf4#357/#358) is now known not to move it on sm_120.

## Cost

Five ledger rows, **$0.7539**, against a $3 lane ceiling and a $5 hard stop. Two of those
rows are this lane's own faults: `p42-census-2` ($0.4853) was my design error, recorded
in Amendment 1, and `p42-census-3` ($0.1147) was a host that refused both account ssh
keys and is now on the exclusion list.
