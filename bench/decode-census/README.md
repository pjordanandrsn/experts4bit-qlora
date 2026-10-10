# Lane 2 design note: where e4b's batched decode step spends its time at 16, 32 and 64 rows

**Zero rental; for review before any registration** (#846, lane 2 of the serving-throughput program). Nothing here is
measured anew. `decode_census.py` computes every number in sections 1–3 from committed receipts, and
`tests/test_decode_census.py` keeps this file equal to its output. The appendix cites SGLang's source and the reading's
server logs.

**Sources.**
- SC5's reading `sc5-5090-2`: e4b 0.52.0 against vLLM 0.31.0, TPOT per cell.
- SC2e's step traces (`verdict_sc2e.json`): the served 16-row step and the prefill stall.
- Three eager kernel-class profiles:
  - P119 at 16 rows;
  - P124 at 32 rows, with int4 attention projections on (the 0.51.0 default);
  - P126 attempt 1 at 64 rows, with the four-program tile table (the 0.52.0 default).
- Each column comes from its own RTX 5090. Ratios inside a column are within one box; columns are not.

## 1. The decode step, by kernel class

| class (device ms per step, eager profile) | 16 rows | 32 rows | 64 rows |
|---|---:|---:|---:|
| int4 experts (K19 grouped GEMM) | 4.61 | 5.91 | 6.91 |
| attention (fp8 paged decode, append) | 0.79 | 1.11 | 1.69 |
| int4 attention projections | 0.78 | 0.82 | 1.45 |
| routing and tile table | 0.64 | 1.11 | 0.80 |
| elementwise | 0.51 | 0.52 | 0.87 |
| dense GEMM (output head) | 0.56 | 0.64 | 0.61 |
| norms and glue | 0.27 | 0.28 | 0.32 |
| other (sampling, copies, misc) | 0.06 | 0.07 | 0.08 |
| **eager device total** | **8.22** | **10.47** | **12.74** |
| served (captured) step, median | 9.56 / 9.65 | 11.39 / 11.40 | 15.36 / 15.47 |
| **residual** = served − eager total | 1.33 / 1.43 | 0.92 / 0.92 | 2.62 / 2.73 |

**Reading the table.**
- The residual closes each column. It is host time and the difference between a captured step at a growing context and
  an eager profile of a few steps.
- At 16 rows, SC2e's own traces put the host part at 1.16 ms: step minus device, draw 1.
- At 64 rows, int4 experts are 54% of the eager step, attention 13% and int4
  projections 11%.
- From 16 to 64 rows the eager step grows 0.094 ms a row, of which the expert GEMM is 0.048.
- At 32 rows the tile table is P124's one-program table. 0.52.0's `E4B_INT4_TILE_PROGRAMS=auto` covers that size too,
  but no lane has read it there.

## 2. SC5's TPOT, split

Closed-loop TPOT is the served decode step plus the prefill stall that every admitted request puts on the decoders.
- SC2e measured that stall at 40.50 ms per admitted 512-token prompt: the median of four servers' direct stall.
- Over a request's 256 tokens, the other C − 1 slots each admit one request, so the stall per token is
  (C − 1) × 40.50 / 256 ms.

| SC5 cell | e4b TPOT p50 | served decode step | prefill stall per token | **residual** | vLLM TPOT p50 |
|---|---:|---:|---:|---:|---:|
| C = 16, default | 11.74 | 9.60 | 2.37 | **-0.24** | 9.00 |
| C = 16, matched | 11.77 | 9.60 | 2.37 | **-0.21** | 8.99 |
| C = 64, matched | 25.53 | 15.42 | 9.97 | **+0.14** | 17.16 |

**What the split shows.**
- The residual closes each row within about 0.2 ms.
- At C = 64, the prefill stall is 39% of e4b's TPOT. e4b's 64-row decode step
  (15.42 ms) is below vLLM's whole TPOT (17.16 ms).
- **Inference, not measured.** Per 64 tokens, e4b's served work is 26.65 ms (64 ÷ its output tok/s) against
  vLLM's 18.41 ms.
  - If vLLM's 64-row decode step were e4b's, vLLM would spend about 2.99 ms per 64 tokens on
    admissions, against e4b's 9.97.
  - That would make e4b's 512-token prefill forward roughly 3.3× vLLM's.
  - No vLLM step-level or prefill-level data exists at these sizes. Only SC1b's 16-row class census does, on a 400 W
    card. A vLLM step census at 64 rows, or a prefill timing at 512 tokens, would settle it.
- **The trade-off SC5 shows** is consistent with this. Against vLLM, e4b leads TTFT in 10 of
  12 cells and trails TPOT at C = 16 default, C = 16 matched, C = 64 matched.

## 3. What this suggests for lane 2 (for review; nothing is registered)

1. **The prefill forward per admitted request.** It is about 9.97 ms of every 64-row TPOT and largest at
   high concurrency. The first zero-rental step would be the same class census for the 512-token prefill forward (40 ms
   on SC2e's servers), to see whether its experts, attention or glue dominate.
2. **The int4 expert GEMM (K19)** is 6.91 ms of the 64-row step and
   4.61 ms of the 16-row step. Its byte-floor efficiency at 32 and 64 rows is not read:
   that needs the distinct experts per step at those sizes.
3. **Attention and projections** together are 3.14 ms at 64
   rows.

## Appendix: SGLang 0.5.21's decode CUDA-graph cap (zero rental)

SC5's read calls SGLang's C = 64 TPOT (48–51 ms, against about 9 ms at C = 16) not diagnosed. The reading's own logs
and SGLang's source account for it.
- **The source.** In the locked wheel (`sglang-0.5.21-cp312-cp312-manylinux_2_34_x86_64.whl`, sha256 `ac300998…`),
  `sglang/srt/arg_groups/memory_hook.py` lines 100–116 set the default decode graph `max_bs` to 48 on a card with 20 to
  35 GiB at tensor parallel below 4. Its comment names the RTX 5090.
- **The box.** The server info reports `cuda_graph_config.decode.max_bs` 48, with graphs captured for batch sizes 1–48.
- **The logs.** In blocks 3 and 9 of the reading (default and matched), every decode batch of 16 or fewer ran with the
  graph (397 log lines each). Every batch above 48 ran without it (32 each). These are the per-block server logs, kept
  with the private receipts.
- At C = 64, SGLang therefore decodes eagerly at its default. This changes no SC5 label; the read's cells stand as
  measured.
