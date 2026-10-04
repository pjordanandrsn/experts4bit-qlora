# SC2: request-level serving on one RTX 5090. vLLM and SGLang hold the SLO to 8 req/s; e4b's `serve_paged` holds it only at 1 req/s, because each 512-token prefill stalls every running decode (lane SC2 of #846; 2026-10-04)

Pre-registration: [`../../sc2/SC2-PREREG.md`](../../sc2/SC2-PREREG.md) (#1014), with amendment A1 (#1033).

**The instrument.** One request driver (`bench/sc2/sc2_driver.py`) sends one plan to each engine's own OpenAI
`/v1/completions`:
- identical token-id prompts (64 rows of 512 wikitext-2 tokens);
- `max_tokens` drawn from U[64, 256];
- greedy decoding, `ignore_eos`, streaming;
- 16 requests in flight and prefix caching off;
- each engine's own scheduling.

**What it measures.** TTFT, TPOT and SLO attainment: the share of a run's requests that are VALID with TTFT ≤ 1.0 s
and TPOT ≤ 100 ms. Goodput is attainment × rate.

**The workloads.** Q1 serial (24 requests) and Q2 Poisson at 1, 2, 4 and 8 req/s (120 requests each), two draws of
everything.

| run (receipt) | e4b | host (driver; board power limit) | outcome | $ |
|---|---|---|---|---|
| `sc2-prove-1` (adertha-receipts `d29277e`) | `887940e` | AMD EPYC 7B13, 256 vCPU (595.84; 575 W), machine 145701 | HARNESS_ERROR, NOT PROVED: e4b's web stack missing; llama.cpp keep-alive → amendment A1 | 0.922 |
| `sc2-prove-2` (`0eaf6fb4`) | `b7e41a7` (A1) | same machine, 145701 | **PROVED box=E**: four servers, eight smokes, every request VALID | 0.954 |
| `sc2-5090-1` (`ed2a6068`) | `b7e41a7` | AMD EPYC 7C13, 256 vCPU (595.71.05; **400 W**, 3090 MHz), machine 45511 | OK: all five engines driven, **5,060 requests, every one VALID** | 2.349 |

SC2 total: **$4.225 across 3 receipts**. Every run is under the $15 no-ask tier.

**The reading's board.** It ran on the board SC1b's box D used, capped at 400 W. That is below the proofs' 575 W and
below SC1's boxes, so absolute times here are a 400 W 5090's. Every comparison below is between engines on that one
board, in one run.

**Cost against the registered guards.** The PREREG stated its guards as hours × $0.75: proof ≤ $0.94, reading ≤ $2.25.
- **What the box bills:** $0.733/h ($0.644 GPU + $0.089 storage), **plus per-GB download** at $0.0078/GB.
- **What the guards left out:** the download charge. The launcher's own estimate counts it.
- **The overruns:** `sc2-prove-2` came to $0.954 (+$0.014 over its guard) and the reading to $2.349 (+$0.099).
- **What it changes:** neither guard is a rule, and neither overrun changes a reading. Stated so the guard is not
  mistaken for the bill.

**Stack.**
- **e4b `b7e41a7`.** Its package code is the registration's (`887940e`) plus the 0.45.0 version string; nothing else
  changed. e4b serves its NF4 arena with SC1's int4 levers, and the health record confirms them: 48 int4 expert
  layers, 96 int4 attention projections, decode graphs on buckets 1–16, `fuse_qkv` on. The labelled e4b_nf4 row has no
  levers.
- **grouped-nf4-gemm v0.34.1** (`34da93d6`), SC1's pinned stack, not e4b CI's v0.37.0. So `GNF4_PDL` and
  `GNF4_TRAIN_GEMM=auto` are not in play.
- **vLLM 0.30.0 and SGLang 0.5.20** serve `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4`.
- **llama.cpp `552f18f`** serves the Q4_K_M GGUF @ `d5b1d57`.

## The outcome by the registered rule

Full tables: [`receipts/sc2-5090-1/RESULTS-sc2.md`](receipts/sc2-5090-1/RESULTS-sc2.md). Rendered from the box's
`verdict.json`, which `sc2_reduce.py` re-derives identically from the committed run files.

**Capacity ceilings** (the largest rate VALID with attainment ≥ 0.95 in both draws): **vLLM 8, SGLang 8, llama.cpp 2,
e4b_int4 1, e4b_nf4 1** req/s.

| engine | serial p50 TTFT | serial p50 TPOT | attainment at 1 / 2 / 4 / 8 req/s (draw 1, draw 2) | ceiling |
|---|---|---|---|---|
| vLLM | 0.052 s | 3.50 ms | 1.00, 1.00 / 1.00, 1.00 / 1.00, 1.00 / 1.00, 0.98 | **8** |
| SGLang | 0.034 s | 3.27 ms | 1.00, 1.00 / 1.00, 1.00 / 1.00, 1.00 / 1.00, 1.00 | **8** |
| llama.cpp | 0.095 s | 3.06 ms | 1.00, 1.00 / 0.97, 1.00 / 0.27, 0.87 (UNSTABLE) / 0.13, 0.10 | **2** |
| e4b_int4 | 0.269 s (UNSTABLE: 0.293 vs 0.246) | 4.51 ms | 1.00, 0.98 / 0.34, 0.81 (UNSTABLE) / 0.08, 0.10 / 0.02, 0.03 | **1** |
| e4b_nf4 (labelled) | 0.275 s | 9.58 ms | 1.00, 0.95 / 0.11, 0.35 (UNSTABLE) / 0.05, 0.04 / 0.02, 0.03 | **1** |

| prediction | verdict | detail |
|---|---|---|
| Q1 serial: e4b_int4's p50 TTFT ≥ 2 × vLLM's | UNREAD | e4b_int4's serial row is UNSTABLE: its draws' p50 TTFT are 0.293 and 0.246 s, 17 % apart against a 10 % band. Seen, not read: 5.18× |
| Q2 serial: e4b_int4's p50 TPOT / vLLM's in [1.10, 1.35] | UNREAD | the same row. Seen, not read: 1.29 |
| Q3 serial: llama.cpp's p50 TPOT is the lowest of the four lane engines | UNREAD | e4b_int4's serial row is not VALID. Seen, not read: llama.cpp 3.06 ms is the lowest (SGLang 3.27, vLLM 3.50, e4b 4.51) |
| Q4 capacity: vLLM ≥ e4b_int4 ≥ llama.cpp | **REFUTED** | 8 ≥ 1, but 1 < 2 |
| Q5 capacity: e4b_nf4 < e4b_int4 | **REFUTED** | both 1 |
| Q6 every row VALID | **REFUTED** | UNSTABLE rows: e4b_int4 serial and 2 req/s, llama.cpp 4 req/s, e4b_nf4 2 req/s. **No row is INVALID**: all 5,060 requests completed with exactly `max_tokens` tokens and finish `length` |

**No position sentence comes from SC2 alone.** SC1's licence read QUALITY_FAIL. These are measured serving behaviours
of the four stacks as configured, to stand beside SC1's per-arm quality rows.

## What it means

**Every SLO miss in the run is a TTFT miss.** No engine missed the 100 ms TPOT bound at any rate. e4b's p50 TPOT under
full load is about 39 ms (int4) and 64 ms (NF4). The engines part at the first token.

**e4b's capacity is set by its prefill, not its decode.** This analysis is post hoc and descriptive (not registered),
from e4b's own request trace (`E4B_PAGED_TRACE`). The tool is `bench/sc2/sc2_trace.py`; the per-workload output is in
[`receipts/sc2-5090-1/RESULTS-trace.md`](receipts/sc2-5090-1/RESULTS-trace.md).
- **The stall.** Across all 1,008 measured int4 requests, decode time fits
  `decode_s = 6.15 ms × (out_len − 1) + 0.356 s × (other requests' prefills completed during this request's decode)`,
  with R² = 0.985. For NF4 the fit is 19.8 ms and 0.48 s, with R² = 0.971.
- **What one stall costs.** `serve_paged` prefills a 512-token prompt as one chunk per step (`max_prefill_tokens_per_step`
  512). That step stalls every request decoding beside it for about a third of a second on this board.
- **Why decode stops being the cost.** At 4 and 8 req/s each request lives through about 14 other prefills (7.6–11.7
  at 2 req/s). That is roughly 5 s of stall against about 1 s of its own decoding.
- **Why it collapses.** The 16 slots stay full and new requests queue for one: queue wait reaches 22 s at 4 req/s. TTFT
  grows with the queue, from a p50 of 0.28 s at 1 req/s to 11 s at 4 req/s and 17 s at 8 req/s (draw 1).
- **The comparison.** e4b's serial TTFT for the same 512 tokens is 0.27 s, against vLLM's 0.05 s and SGLang's 0.03 s.
  e4b's decode at low load (about 4.5 ms TPOT serial) is within 1.3× of vLLM's. Its prefill is about 5× slower, and
  under load every arrival makes all the decodes pay for it.

**llama.cpp's limit is different.** Its TTFT stays low (0.095 s serial), but its TPOT with 16 slots busy is about 27 ms.
At about 4 s per request, 16 slots serve about 3.6 req/s, so it gives way between the two draws' realised 3.4 and
4.3 req/s. That is the UNSTABLE 4 req/s row.

**Why the knee rows are UNSTABLE.** The two draws use different Poisson seeds, and their realised arrival rates
differ: 2.32 vs 1.89 req/s at nominal 2, and 4.32 vs 3.39 at nominal 4. Near an engine's knee that difference alone
moves attainment from 0.34 to 0.81 (e4b at 2) or from 0.27 to 0.87 (llama.cpp at 4). The rule reads these rows as
UNSTABLE, which is honest. A design that fixes the realised rate per draw, or repeats one realisation, would separate
an engine's knee from the dice. Noted for the next serving lane, not amended here.

**The lever this names for e4b.** Make prefill cheaper, or stop it stalling decode: a smaller prefill chunk per step,
a prefill token budget shared with decode, or a faster int4 prefill path. The decode-side levers (SC1b's KV-select and
PDL) do not touch this. P111's KV-select is in this run, and PDL is not.

## What was fixed before the reading, and why

- **Before registration** (self-review in #1014; no data existed):
  - **Capacity.** It read good requests / wall time, and the wall includes the drain after the last arrival. At 8 req/s
    that caps a perfect engine at about 6.7 req/s, so no ceiling above 2 req/s was reachable. Capacity reads SLO
    attainment instead.
  - **TPOT spread.** A rate row's TPOT spread between draws is reported, not a status. Here it is flagged on 6 rows and
    gates none.
  - **Chunk timing.** The driver times only chunks that carry text. llama.cpp's final chunk is an empty-text
    finish-only chunk (1,008 of its 1,008 requests), and vLLM and e4b held back a partial character 45 and 35 times;
    none of them is timed.
  - **Q3** ranges over the four lane engines.
- **After `sc2-prove-1`** (amendment A1, harness only):
  - e4b's box now installs `serve_paged`'s web stack, pinned.
  - The driver opens a fresh connection per request on every engine. llama.cpp drops a connection after a streamed
    response, and every other request had failed on the reused socket.
  - The e4b start waits for `/health` status `ready`, not HTTP 200.

## Reproduce

- **Rerun the lane:** `SC1_BOX=E bash bench/sc1/sc1_drive.sh`, from a controller at `b7e41a7`, under a launch spec
  guarded on the PREREG, the driver, the reducer and box E.
- **Re-reduce:** `python bench/sc2/sc2_reduce.py --dir receipts/sc2-5090-1/sc2 --out verdict.json`.
- **Re-run the trace analysis:** `python bench/sc2/sc2_trace.py receipts/sc2-5090-1/sc2/trace_e4b_int4.jsonl`.
- **What the committed receipts leave out.** The driver records drop their per-token chunk gaps, which the reducer
  never reads. The 28 MB llama.cpp server log is cut to its head and tail. The prompt pool is left out; its sha256 is in
  every run file. All three are complete in the receipt store's committed receipts.
