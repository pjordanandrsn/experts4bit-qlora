# P148 (bench/p148): the dense bf16 decode GEMVs, a falsify-first microbench (e4b#1313)

**Status:** registered, not run. A microbench. It decides whether two kernel levers get built. **A saving here is a
premise, never the quoted gain:** a GO starts kernel work in grouped-nf4-gemm and a served read of its own. No default,
gate or speed claim changes.

## Why, from data already seen

P142's per-name table (`pername-5090-1`, P = 4.380 ms at one request, Qwen3-30B-A3B on an RTX 5090) has the dense bf16
GEMVs (cuBLAS `kernel`) as the top removable lines after P143's attention cap, 41 % of P:

| row | per step | exclusive time | bytes | DRAM rate |
|---|---|---|---|---|
| fused q/k/v, 5120 × 2048 (grid 1280) | 48 | 0.715 ms | 1007 MB | 1.41 TB/s, 79 % of 1.792 |
| o_proj, 2048 × 4096 (grid 512) | 48 | 0.597 ms | 805 MB | 1.35 TB/s, 75 % |
| lm_head, 151936 × 2048 (grid 37984) | 1 | 0.377 ms | 622 MB | 1.65 TB/s, 92 % |

q/k/v and o leave about 15 % of the bandwidth unused (o runs 512 four-row blocks on 170 SMs). The w8a16 split premise
(e4b#1745) **reported** an int8 lm_head as nearly free on quality (t1 argmax 0.989, d +0.0005). That was reported, not
ruled.

Two levers:
- **D1:** a bandwidth-targeted bf16 GEMV for q/k/v and o at T == 1: grouped-nf4-gemm K33's design (`_gemv_nf4_bw`)
  without the NF4 decode. At 90 % of peak it would save about 0.19 ms. It needs no quality bar beyond a tolerance
  contract: the same bf16 weights and fp32 accumulation, only regrouped sums.
- **D2:** an int8 lm_head (W8A16, per-row scale), half the bytes, about 0.19 ms. It carries a quality obligation
  (below).

## The kernels (`p148_kernels.py`)

Prototypes, not library kernels.
- **`gemv_bf16`:** program `(pid_n, pid_k)` owns `BLOCK_N` rows over a K span, loads `[BLOCK_N, KC]` tiles and
  accumulates in fp32. `split_k > 1` stores fp32 partials that the host sums in split order (K33's split-K contract).
- **`gemv_i8`:** int8 rows times an fp32 per-row scale (`quantize_rows_int8`: symmetric absmax / 127), fp32
  accumulation.
- **Registered grids:**
  - bf16 `(BLOCK_N, KC, warps, split_k)` ∈ {2, 4, 8} × {512, 1024} × {4, 8} × {1, 2, 4}: 36 configs;
  - int8 `(BLOCK_N, KC, warps)` ∈ {4, 8, 16} × {512, 1024, 2048} × {4, 8}: 18 configs.
- **Their own suite** (`tests/test_p148_kernels_interp.py`, 61 tests):
  - every config at ragged shapes against fp64;
  - a one-hot activation reads one column exactly;
  - split-K agrees with the unsplit kernel;
  - the int8 kernel reads its dequantized weights;
  - the quantizer's range.

  CI runs it under Triton's interpreter. The box runs it **compiled on the card** after P136's premise, and must read
  61 passed, none skipped.

## The bench (`p148_bench.py`)

One no-model process. Every launch reads its own weight tensor, as the served step does: q/k/v 48 × 21 MB, o 48 ×
16.8 MB and the head 622 MB, all well above the 96 MB L2. cuBLAS is `torch.nn.functional.linear` on a `[1, K]` row, as
`nn.Linear` serves it, in the **same** graph-captured harness as the kernels.
1. **Contract:** per shape, layer 0's weight and 4 random rows. Relative RMS error against an fp64 reference, for cuBLAS
   and for every config (D2: the int8 kernel against its dequantized weights).
2. **Select:** per shape, every config whose error is ≤ 1.05 × cuBLAS's. One graph of the shape's launches, 10 warm +
   50 timed replays (median); the fastest is the choice.
3. **Read:** per shape, fresh graphs A (cuBLAS) and B (the choice), timed **A1, B1, B2, A2**, 20 warm + 200 timed
   replays each, each replay by its own event pair, medians. The pairs (A1, B1) and (B2, A2) run in opposite orders.

## The rule (`p148_reduce.py`, fixed before data)

Every bar compares unrounded values. BAR = 2 % of P142's P = **0.087603 ms**.

1. **VOID** if any of these holds:
   - the record is missing or not ok;
   - the card is not a 170-SM RTX 5090;
   - a shape's contract, selection or read is missing;
   - a choice is not the fastest contract-passing config of its own selection;
   - **the harness does not transfer**: a shape's cuBLAS medians (A1, A2) lie outside [0.8, 1.25] × P142's in-situ time
     (q/k/v 0.7146, o 0.5972, head 0.3770 ms).
2. **NOISY** (a VOID, and a strike): a shape's self-pair, A2/A1 or B2/B1, lies outside [0.985, 1.015].
3. Then each lever independently. A shape's saving is the **smaller** of its two pairs, A1 − B1 and A2 − B2.
   - **D1: GO** iff q/k/v and o each have a contract-passing config and their savings sum to ≥ BAR. Otherwise **PARK**.
   - **D2: GO** iff the int8 head kernel passes its contract and its saving is ≥ BAR. Otherwise **PARK**.

The verdict is `D1_<GO|PARK>+D2_<GO|PARK>`.

**Reported, never ruling:** every config's error and selection time, each shape's DRAM rate against 1.792 TB/s, and
the pairs.

**After a GO:**
- **D1** becomes a grouped-nf4-gemm kernel with its own tolerance contract, then a served read (P143's form).
- **D2** carries a **quality obligation** first. Its served registration must rule on quality with the split premise's
  rule at **both** gates (t1 and k16: argmax ≥ 0.95 and d + 1.645 SE ≤ the bound) before anything flips. The earlier
  I8HEAD numbers were reported, not ruled.

**Predictions (graded, never ruling):**
- D1: the best bf16 configs reach 85–92 % of peak at q/k/v and o, saving 0.08–0.18 ms in total; D1_GO;
- D2: the int8 head saves 0.15–0.20 ms; D2_GO;
- every self-pair within [0.99, 1.01].

## The box

- **Run ids:** `p148-5090-1`, with `-2` and `-3` as pre-create spares. A verdict after the bench is the result.
- **Card and budget:** one RTX 5090 (cc 12.0, ≥ 30 GB, 170 SMs) on torch 2.8.0+cu128, at ≤ $0.80/h.
  - Disk: 60 GB (the runner refuses below 40 GB).
  - Download: ≤ 5 GB of pip, at ≤ $0.005/GB.
  - Guard: 0.5 h. Worst case about **$0.43**, inside the maintainer's ~$0.45 (02:58Z).
- **Runner** (`p148_run.sh`): P143 step 1's runner, by counted edits. Its refusals, install (e4b at this registration's
  merge, grouped-nf4-gemm 0.47.1 @ `38b30888`), tripwire and P136's 19 premise tests are kept. Then the kernels' suite
  runs compiled, then the bench, then the reducer.
- **Driver** (`p148_drive.sh`): P143 A3's, by counted edits, on `bench/common/lane_liveness.sh` (launch PID + `/proc`
  identity). No model, so no hf token.
- **Approval:** the maintainer's GO (bus 02:58Z) under the standing rule. One review by any seat that does not consume
  this work, then the maintainer's final pass.
