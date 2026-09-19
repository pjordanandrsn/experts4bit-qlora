# Results — K16 P5: the attention GEMM row with the small-M route on (Qwen3-30B-A3B, B=16, RTX 5090, 2026-09-19)

Registered in grouped-nf4-gemm `kernel/PREREG-k16-smallm-int4-gemm.md` (P5 + amendment 1); runner `bench/k16/k16p5_run.sh`, reducer `bench/k16/k16p5_reduce.py`. Run `k16-p5` (vast instance 51528841, receipt `2026-09-19/k16-p5/`; e4b `29bbc41` with the opt-in route #578, gnf4 `f189e67` — the K16 lane's cut, the bytes the 5090 kernel read measured). P42's protocol: `step_decomp.py --batch 16 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --placement-override all-vram`, 70 timed steps after 3 warm, `--replay-profile-out` = 8 profiled replays AFTER the timed window (the timed number is byte-identical to a no-census run). Three arms on one box, in order: `nf4_b16` (control), `int4_b16` (P42's arm: RTN int4 experts + uncalibrated int4 attention, rows > 1 on `Int4Linear`'s cached-bf16 matmul), `int4_b16_smallm` (the SAME bytes with `E4B_ATTN_INT4_SMALLM=1`).

## The rows

| arm | timed step (ms, clean) | agg tok/s | bf16 GEMM family ms/step (kernels) | `_gemm_int4_b32_smallm` ms/step (calls/step) | census kernels |
|---|---|---|---|---|---|
| `nf4_b16` | 32.24 | 496.2 | 25.623 (5) | 0 (0) | 51 |
| `int4_b16` | 12.25 | 1306.5 | 2.917 (4) | 0 (0) | 38 |
| `int4_b16_smallm` | **11.19** | 1429.3 | 0.565 (4) | 1.302 (192) | 39 |

Proof the route engaged where it should and nowhere else (`K16ROUTE` line in `summary.txt`): `_gemm_int4_b32_smallm` appears in the smallm arm's census (192 calls per step = 4 projections × 48 layers) and in no other arm's. The bf16 GEMM family (`cutlass … wmma_tensorop_bf16` + `cublasLt splitKreduce`) that carried the attention projections in `int4_b16` is what falls.

## P5, read against the registered rule

**P5 HOLDS.** The bf16 GEMM family falls by **+2.352 ms/step** from `int4_b16` to `int4_b16_smallm` (threshold ≥ 0.4); the K16 kernel costs **1.302 ms/step** in its place, a net attention-GEMM change of **+1.050 ms/step**; the timed step moves **12.25 → 11.19 ms (−1.06 ms, −8.6 %)**, aggregate throughput 1,306 → 1,429 tok/s (+9.4 %). The net saving matches the census within 0.01 ms — the route did exactly what the census says it did and nothing else moved (every other top kernel row is within noise between the two int4 arms: `_gemv_int4_b32` 6.28 vs 6.37, `_fp8_paged_decode_split_f8dot` 0.70 vs 0.68, `_reduce_partials` 0.33 vs 0.33).

Kernel-level context (5090, K16 lane): per launch the route replaces a 10.37 / 6.23 / 6.29 / 18.61 µs bf16 GEMM with 6.35 / 4.37 / 4.49 / 6.39 µs — 19.9 µs saved per layer per step × 48 layers = **0.96 ms/step predicted from the microbench**; the census reads **1.05 ms/step net**, within 0.1 ms. The bf16 family's census level in `int4_b16` (2.92 ms) is above the 48 × 41.5 µs = 1.99 ms the four projections account for because the same kernel family also carries the lm_head and router matmuls, present in both arms — the DROP is the number, not the level.

## Decision

P5 holds at the model level, so the consumer route earns its default: **`E4B_ATTN_INT4_SMALLM` defaults to `auto`** — on when the installed grouped-nf4-gemm carries `int4_smallm` (≥ 0.32.0), off with a one-line notice in the enable banner otherwise (never a silent fallback and never a refusal on an older cut, which is what an opt-in `=1` does). `=1` still refuses without the kernel; `=0` keeps the cached-bf16 path. The B=1 path (the int4 GEMV) is untouched, so every K8 row on record is unaffected; the route computes the same `x_bf16 @ dequant(W)` with fp32 accumulation the cached-bf16 matmul computes, within bf16 rounding (K16's contract, 0.003 relative on this card).

## Cost

`k16-p5`: ≈ 20 min of a $0.65/h box ≈ $0.22 against a $0.98 estimate. Torn down with proof.
