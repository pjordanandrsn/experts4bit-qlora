# SC2g: on gpt-oss-20b, e4b's `serve_paged` serves every row VALID and is prefill-bound under load, as registered (Q4 holds, b/a 27); capacity is 1 req/s against vLLM's and SGLang's 8, and its single-stream decode is 1.74× vLLM's (lane SC2g of #846; 2026-10-05)

Pre-registration: [`../../sc2/SC2g-PREREG.md`](../../sc2/SC2g-PREREG.md) (#1095, reviewed by the maintainer before it was
registered; amendment A1, #1121). This is e4b's first gpt-oss run through `serve_paged`. Every e4b-vs-comparator ratio
here is **ARITH_MISMATCH**: the engines serve the same checkpoint's MXFP4 experts on different arithmetic.

| run (receipt) | e4b | host (driver; board power limit) | outcome | $ |
|---|---|---|---|---|
| `sc2g-prove-1` (adertha-receipts `ae066e41`) | `e8892972` | machine 26157 | **HARNESS_ERROR**: box G died sourcing its own script under `set -u` (`FOLDS: unbound variable`); fixed by A1 | 0.848 |
| `sc2g-prove-2` (`8c13ef70`) | `eaf3e5b4` | AMD EPYC 7C13, 256 vCPU (400 W, 3090 MHz), machine 45511 | **PROVED box=G**: e4b check ok; four servers on gpt-oss, all 8 smokes VALID | 1.245 |
| `sc2g-5090-1` (`dcd6740d`) | `eaf3e5b4` | machine 45511 | NOT_RUN: the launcher's pre-flight could not reach the box (ssh refused); nothing ran | 0.000 |
| `sc2g-5090-2` (`2f70d89a`) | `eaf3e5b4` | AMD Ryzen Threadripper PRO 3955WX, 32 threads (590.48.01; **575 W**, 3090 MHz), machine 26157 | OK: four engines, both draws, every arm | 0.959 |

SC2g total: **$3.052 across 4 receipts**, inside the registered ~$3.4. The A1 re-proof is part of that.

**Stack.**
- e4b `eaf3e5b4`. Its serving code is the registration's `e8892972`; A1 touched only `bench/` and `tests/`.
- grouped-nf4-gemm v0.41.0 (`dc8f94ab`), `GNF4_TRITON_PREBIND=1`.
- openai/gpt-oss-20b @ `6cee5e81`, and ggml-org's `gpt-oss-20b-MXFP4.gguf` @ `ef9b12f2` for llama.cpp.
- vLLM 0.30.0, SGLang 0.5.20, llama.cpp `552f18f`.
- One CUDA 13 image. 512-token prompts from gpt-oss's own tokenizer: all 1,012 of e4b's requests reported 512.

**The board.** The reading ran on a 575 W board, not SC2's and SC2b's 400 W one (machine 45511). Absolute times are not
comparable across the two. Everything below compares engines within this run.

## What e4b ran, and how that is known

**Q4 and this section rest on the code path, not on `/health`'s route names.** The maintainer checked this against the
code at the run commit, and I verified it. `/health`'s `prefill_routes` is resolved from the environment
(`serve_paged.py:651`), so on gpt-oss its "k19 / k19 / flash" does **not** describe the path.

**The path gpt-oss takes:**
- **Decode.** On the MXFP4 store with device grouping, rows ≤ 256 take **K21** (`gemm_mxfp4_grouped_smallm`, bf16
  activations, W4A16; `hot_residency.py:446`). T == 1 stays on the **GEMV on int8 activations** (W4A8).
- **Prefill.** A 512-token chunk is 2,048 expert rows. Rows above 256 take the **kept NF4 stacks' M-tile GEMM**
  (`hot_residency.py:462`). The NF4 stacks are emptied unless `E4B_INT4_KEEP_NF4=1` (`int4_experts.py:616`).
- **Attention.** Every gpt-oss layer has sinks, so prefill attention stays on the **explicit-mask path** on all 24
  layers (`paged_attention.py:327`).

**What was observed:**
- `int4_store_kinds` `["mxfp4"]` and `int4_expert_layers` 24, at ready.
- `levers_env` records `E4B_INT4_KEEP_NF4=1`.
- VRAM at ready was 26,316 of 32,607 MiB. That is *consistent with* both expert stores being resident; it does not
  observe them. No log line or counter says the NF4 stacks were kept.

**Box G's check passed its route assertion vacuously.** The assertion was copied from box F's Qwen3 check, so it gated
the store kind and the layer count, not the route. e4b#1129 (merged while this run was reading, not in its commit) adds
`prefill_routes.seen`, the route each call actually took.

**The prefill graph's `auto` engaged** at ready: status "on", T 512, pool 246 MiB, 5,796 MiB free after. The
registration named a memory stand-down as plausible; it did not happen.
- The reading recorded `/health` at ready only, so its replay count is not in the receipt.
- Every request was a single 512-token chunk, which is the replay condition (`paged_runner.py:172`).
- The proof's end record read replays = requests (22 / 22, 0 eager chunks).

**SGLang** resolved its default MoE runner to **`flashinfer_mxfp4`** (`server_info`), with triton attention and radix
off. On sm_120 that runner is cutlass_sm120 with MXFP8 activations, i.e. **W4A8**
(`finding_gptoss_mxfp4_serving_arithmetic_differs_per_engine`, read at the pins).

*Footnote:* the box's KNOBS line prints SC1's common Qwen3 constants on box G. It is not the served model: `/health`
reads `gpt_oss` @ `6cee5e81`.

## The outcome by the registered rule

Full tables: [`receipts/sc2g-5090-2/RESULTS-sc2g.md`](receipts/sc2g-5090-2/RESULTS-sc2g.md). `sc2g_reduce.py` re-derives
`verdict_sc2g.json` identically from the committed run files.

| engine (arithmetic) | serial p50 TTFT | serial p50 TPOT | attainment at 1 / 2 / 4 / 8 req/s (draw 1, draw 2) | ceiling |
|---|---|---|---|---|
| e4b_gptoss (MXFP4 decode W4A8 / W4A16; NF4 prefill) | 0.166 s | 6.21 ms | 1.00, 1.00 / 0.81, 0.88 / 0.18, 0.18 / 0.12, 0.10 | **1** |
| vLLM (Marlin W4A16) | 0.033 s | 3.56 ms | 1.00 at every rate, both draws | **8** |
| SGLang (`flashinfer_mxfp4`, W4A8) | 0.024 s | 4.06 ms | 1.00 / 0.97, 1.00 / 1.00 / 1.00 | **8** |
| llama.cpp (GGUF, attention Q8_0; W4A8 decode, W4A4 prefill) | 0.048 s (**UNSTABLE**) | 3.03 ms | 1.00 / 1.00 / 0.49, 0.47 / 0.17, 0.17 | **2** |

| prediction | verdict | detail |
|---|---|---|
| Q1 serial p50 TTFT e4b ≥ 2 × vLLM | **HOLDS** | 5.05× (0.166 s against 0.033 s); 6.97× against SGLang |
| Q2 serial p50 TPOT e4b ≤ 1.5 × vLLM | **REFUTED** | 1.74× (6.21 ms against 3.56 ms); 1.53× against SGLang; 2.05× against llama.cpp's means (its row is UNSTABLE on TTFT; its TPOT read 3.03 ms in both draws) |
| Q3 vLLM's ceiling ≥ 4 req/s and > e4b's | **HOLDS** | vLLM 8, e4b 1 (SGLang 8, llama.cpp 2) |
| Q4 e4b's trace fits `a·(out_len−1) + b·(prefills during decode)` with R² ≥ 0.9 and b ≥ 10·a | **HOLDS** | a 8.94 ms per token, **b 0.245 s per prefill**, R² 0.984, b/a **27.4**, n 1,008 |
| Q5 every row of every engine VALID | **REFUTED** | 19 of 20 rows VALID; llama.cpp's serial row is UNSTABLE: p50 TTFT 50.9 ms against 44.5 ms on identical seeds |

## What it means

**The SC2 mechanism holds on a second family, and this time it was registered in advance.** e4b's decode under load is
stalled by other requests' prefills: 0.245 s per prefill landing during a decode, against 8.94 ms per own token. In
SC2 (Qwen3-30B-A3B) it was found post hoc: b/a 58 there, about 44 in SC2b's graphed arm. gpt-oss's b/a of 27 is
smaller, because its per-token decode under load is slower and its stall shorter, but it clears the bar of 10 almost
threefold. **As on Qwen3, most of the stall is outside the graphed forward.** The stall (0.245 s) is about 1.5× the
whole serial TTFT (0.166 s, with the graph on). SC2b read 1.6× on Qwen3. That is the census SC2c now targets.

**Capacity follows from it.**
- e4b holds 1 req/s; at 2 req/s it reached 0.81 and 0.88.
- Its p50 TTFT climbs from 0.18 s at 1 req/s to 5.9 s at 4 req/s. Its p50 TPOT climbs from 10 ms to 31 ms, which is
  under the 100 ms SLO, so the TTFT tail is what fails.
- vLLM and SGLang hold all four rates. Their p50 TTFT stays at 24–47 ms and their p50 TPOT at about 9 ms even at
  8 req/s.

**Single-stream decode is a second gap, and Q2 got it wrong.**
- e4b's served B=1 TPOT is 6.21 ms. The registration's bare-window figures, from other hosts, were 151.1 and 173.3 tok/s
  (6.62 and 5.77 ms per token). Through `serve_paged` the decode lands between them, so the server adds little to it.
- What Q2 underestimated is the comparators. vLLM's Marlin W4A16 decodes in 3.56 ms, SGLang's W4A8 runner in 4.06 ms,
  and llama.cpp's MMVQ in 3.03 ms.
- The registration called Q2 a near coin flip because vLLM's gpt-oss B=1 on this card was unmeasured. It is now
  measured.

**The repeated realisation worked where it was aimed.** All 16 Poisson rows (4 rates × 4 engines) are VALID,
including the knees (e4b at 4 req/s: 0.18 and 0.18; llama.cpp: 0.49 and 0.47). SC2's UNSTABLE knee rows came from
per-draw seeds, and here they are gone. Q5 still fails, on one serial row. llama.cpp's p50 TTFT moved 6 ms (51 → 44 ms)
between two runs of the same 24 requests, and its draw-2 p99 was 156 ms. That is the engine and host, not the dice. The
rule is unchanged, so Q5 is REFUTED.

**No position sentence.** As registered, this measures four stacks as configured, with their arithmetic labelled. There
is no quality claim; that is SC1g's question.

## Next

- **The lever is still the per-prefill work outside the forward**, now on both families. SC2c (#1132; bulk KV
  bookkeeping, another session's lane) is aimed at exactly that; gpt-oss is the natural second model for it.
- **gpt-oss single-stream decode** (6.21 ms against 3.0–4.1 ms) is a separate gap. The B=1 route is the int8-activation
  GEMV.
- **Box G's engagement check** should assert e4b#1129's `seen` fields:
  - `seen.moe` holds `mxfp4_k21|le256` and `nf4_mtile_captured|gt256`, and no `mxfp4_*|gt256` (KEEP_NF4 observed);
  - `seen.prefill_attn` is all `explicit_mask:sinks`;
  - `/health` should also be recorded at the end of the reading, not only at ready.
- **The controller fix in A1 is load-bearing beyond SC2g.** LANE DEAD had never been able to fire on any box.

## Reproduce

- **Rerun:** `SC1_BOX=G bash bench/sc1/sc1_drive.sh` at `eaf3e5b4`.
- **Re-reduce:** `python bench/sc2/sc2g_reduce.py --dir receipts/sc2g-5090-2/sc2 --out verdict_sc2g.json`.
- **Re-fit the stall:** `python bench/sc2/sc2_trace.py receipts/sc2g-5090-2/sc2/trace_e4b_gptoss.jsonl` (its default
  plan is box G's: 4 warm, then serial 24 and 120 per rate, twice).
- **What the receipts leave out:** the driver records' per-token chunk gaps and the prompt pool. Both are complete in
  the receipt store.
