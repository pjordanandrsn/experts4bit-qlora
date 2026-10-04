# SC2: request-level serving. e4b's `serve_paged` against vLLM, SGLang and llama.cpp's server behind one OpenAI endpoint, under Poisson arrivals, on one RTX 5090 (lane SC2 of #846; registered 2026-10-04, before any run)

Claimed by branch `sc2/prereg` (pushed 2026-10-04) and on #846. Follows SC1 (`bench/h2h-2026-10-02/sc1/`) and SC1b
(`bench/h2h-2026-10-02/sc1b/`).

## Why this lane

SC1 timed each engine's decode loop at a fixed batch on identical token ids, and SC1b explained the gaps kernel by
kernel. Neither is how a server is used. In serving, requests arrive at random, prompts prefill while others decode, and
what matters is time to first token (TTFT), the per-token pace (TPOT), and how much offered load an engine serves within
a latency budget (goodput). SC2 asks those questions of the same four engines, each behind its own OpenAI-compatible
`/v1/completions`, driven by ONE client with ONE request plan.

e4b's server is `experts4bit_qlora.serve_paged`, the module the campaign built for exactly this. It takes token-id
prompts, streams one SSE chunk per token, and honours `ignore_eos` and `stop_token_ids`. Its decode graphs have been on
by default since P109.

## Engines (box E: one box, SC1's stack and checkpoints)

| engine | server | weights | settings SC2 pins |
|---|---|---|---|
| **e4b_int4** | `python -m experts4bit_qlora.serve_paged` | NF4 arena baked on the box, SC1's int4 levers (`SPEEDENV`, the three folds, `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`) | `max_seqs` 16 (the default), `E4B_PAGED_MAX_TOKENS_PER_SEQ=2048`, every other route at main's default |
| **vllm** | vLLM 0.30.0 `vllm.entrypoints.openai.api_server` | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4` (GPTQ Marlin) | `--max-num-seqs 16 --max-model-len 2048 --no-enable-prefix-caching --seed 0 --gpu-memory-utilization 0.90` |
| **sglang** | SGLang 0.5.20 `sglang.launch_server` (SC1's `native` mode) | the same GPTQ | `--disable-radix-cache --max-running-requests 16 --dtype float16 --context-length 2048 --mem-fraction-static 0.75` |
| **llamacpp** | llama.cpp `552f18f` `llama-server` (SC1's flags) | `unsloth/Qwen3-30B-A3B-GGUF` Q4_K_M @ `d5b1d57` | `-np 16 -c 16384 -ngl 99 -fa on` (1,024 tokens per slot), `cache_prompt: false` on every request |
| e4b_nf4 (labelled row) | `serve_paged`, the shipped default | the NF4 arena, no levers | as e4b_int4 without the levers |

**The image** is SC1 box C's `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, the one image that holds all four engines.

**Workload matched, scheduling native.** Every engine is offered the same requests at the same instants: identical
token-id prompts, identical `max_tokens`, greedy decoding, `ignore_eos`, streaming. Each engine's own scheduler
otherwise runs at its defaults (chunked-prefill size, policy, CUDA-graph sizes). The only exceptions:
- concurrency is capped at 16 on every engine;
- the sequence window is 2,048. llama.cpp keeps SC1's `-c 16384` over 16 slots, 1,024 per slot, which holds every
  request (at most 512 + 256 = 768 tokens);
- prefix caching is off everywhere. The prompts are distinct, so caching would not hit anyway; it is off so that is
  certain.

**e4b's routes.** e4b serves at main's current defaults:
- int4 prefill through k19 (P102);
- flash prefill attention (P107);
- decode graphs on (P109/P110);
- one KV-table selection per decode step (`E4B_KV_STEP_SELECT`, on by default since 0.44.0; P111).

SC1's two route pins (`loop`, `math`) existed so SC1's own boxes compared alike; SC2 is a fresh lane and drops them.

## The instrument

**The driver** is `bench/sc2/sc2_driver.py`, run unchanged against each engine.
- **The plan.** A fixed request plan, deterministic in (mode, rate, n, seed). Each request has an arrival offset, a
  prompt and a `max_tokens` drawn uniformly from [64, 256].
- **Q1, `serial`:** one request in flight at a time.
- **Q2, `poisson`:** open-loop exponential inter-arrivals at rate r.
- **Every request** streams with `stream_options.include_usage`.
- **Per request:**
  - a token chunk is a chunk whose choice carries text. A choice chunk with empty text (a finish-only chunk, or a
    partial character the detokenizer holds back) is counted but never timed;
  - TTFT is the time to the first token chunk;
  - TPOT is (last token chunk − first token chunk) / (tokens − 1), which stays per-token when an engine packs several
    tokens into a chunk;
  - E2E latency;
  - the streamed text.
- **Validity.** A request is VALID iff it got HTTP 200, the server reported exactly `max_tokens` completion tokens, and
  it finished `length`.
- **SLO attainment** is the share of a run's requests that are VALID and within the SLO: **TTFT ≤ 1.0 s AND TPOT ≤ 100 ms**,
  stated now. **Goodput** is attainment × r. Attainment, not good requests per second of wall time, is the capacity
  quantity: the wall includes the drain after the last arrival, which caps good requests / wall below r even for a perfect
  engine (at 8 req/s, 120 arrivals span about 15 s and the drain adds about 3 s, so at most about 6.7 req/s).

**The prompts** are `bench/sc2/sc2_prompts.py`: 64 distinct rows of 512 tokens of wikitext-2-raw test (row k from token
k·2048), tokenised once with the bf16 checkpoint's tokenizer. Every engine receives these ids.

**Per engine, in order:**
1. start the server and record its health or server info (engagement);
2. 4 warm serial requests;
3. **two draws** of each of these, with draw-specific seeds:
   - Q1 serial, 24 requests;
   - Q2 Poisson at **r = 1, 2, 4 and 8 req/s**, 120 requests each;
4. stop the server and confirm the GPU is empty.

The engine order is e4b_int4, vLLM, SGLang, llama.cpp, e4b_nf4. The deadline drops arms from the end. SC1's sampler
records SM clock, power and temperature throughout.

**Tested on CPU** (`tests/test_sc2_driver.py`, in CI):
- the driver's exact request body through `serve_paged`'s real app, with the streamed bytes going through the driver's
  own parser;
- over real sockets (aiohttp): one chunk per token, packed chunks, an early stop, an HTTP 500, and the Poisson schedule.

## The rule (`bench/sc2/sc2_reduce.py`, self-tested on 9 cases)

A **row** is one engine at one workload (`serial`, or one rate). Its status is the first that applies:
- **UNREAD:** a draw is missing (the deadline, a skipped phase, a server that did not start).
- **INVALID:** any request in either draw is not VALID.
- **UNSTABLE:** the draws disagree.
  - serial: p50 TTFT beyond 10 %, or p50 TPOT beyond 5 %;
  - a rate: p50 TPOT beyond 10 %, or attainment beyond max(10 % of the larger, 0.05).
- **VALID:** otherwise.

**An engine's capacity ceiling** is the largest rate whose row is VALID with attainment ≥ 0.95 in both draws. It is 0
if none qualifies, and UNREAD if any rate's row is UNREAD or INVALID.

**Reported, with no bar:** every row's draw means, and e4b_int4 against each comparator: the serial TTFT and TPOT ratios,
and the goodput ratio at each rate.

**No position sentence comes from SC2 alone.** SC1's licence read QUALITY_FAIL, so SC2's ratios are measured serving
behaviour of the four stacks as configured. They stand beside SC1's per-arm quality rows, never as a quality claim.

## Predictions (written before any data)

| # | prediction | basis |
|---|---|---|
| Q1 | serial: e4b_int4's p50 TTFT ≥ 2 × vLLM's | P102: a k19 512-token prefill is about 0.11 s; vLLM's is a few tens of ms |
| Q2 | serial: e4b_int4's p50 TPOT / vLLM's lies in [1.10, 1.35] | SC1: 1.17–1.20 at B=1 |
| Q3 | serial: llama.cpp's p50 TPOT is the lowest of the four lane engines (the labelled NF4 row is not in it) | SC1: llama.cpp led B=1 decode (1.48× e4b) |
| Q4 | capacity: vLLM's ceiling ≥ e4b_int4's ≥ llama.cpp's | SC1's B=16 decode (vLLM 8.2–8.8 ms, e4b 9.6–10.4, llama.cpp 15.4) plus prefill cost under load |
| Q5 | capacity: e4b_nf4's ceiling < e4b_int4's | P109: the NF4 default runs a 21.6 ms B=16 step |
| Q6 | every row of every engine is VALID | |

## Proof (`sc2-prove-*`, guard 1.25 h ≤ $0.94)

**What it runs:**
- SC1's common proof: installs, tripwires, the Granite fetch and bake, and the Granite smokes;
- the SC2 driver's and reducer's self-tests;
- the SC2 prompt pool, from Granite's tokenizer;
- **every server, on the real lane checkpoints for the comparators**: the GPTQ for vLLM and SGLang, the Q4_K_M GGUF for
  llama.cpp, and e4b on Granite. Each must answer a 6-request serial smoke and a 16-request Poisson smoke at 4 req/s,
  with every request VALID.

PROVED requires every server's smokes to pass. A server that fails is amended before the reading.

## Budget

- **Proof:** $0.94.
- **Reading** (`sc2-5090-*`): guard 3.0 h ≤ $2.25 at $0.75/h.
  - Expected about 2–2.5 h: installs about 15 min, fetches and bake 25–60 min, five engines at about 14 min each.
  - That makes the expected lane cost about $2.5. The two guards together come to $3.19, $0.19 over the campaign's $3
    planning line for SC2. Both runs are inside the $15 no-ask tier.
- The deadline drops the labelled NF4 row first.

## Out of scope

- The prompt-length sweep and longer prompts (a later lane).
- Prefix-cache hits; sampling; chat templates (completions only).
- Fairness, priorities, preemption (e4b has none).
- gpt-oss (SC1g).
- Rates beyond 8 req/s.
- Quality, which is SC1's.
