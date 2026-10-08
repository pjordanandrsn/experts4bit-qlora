# Serving over HTTP (Docker)

*(Moved out of the top-level README for length; linked from it.)* [← back to README](../README.md)

`experts4bit_qlora.serve` wraps the inference path in a FastAPI app so the fine-tune can be
shared by other services instead of each caller paying its own model load — built for a small
GPU that has other tenants. With `OFFLOAD_EXPERTS=1` (the serve default) the OLMoE endpoint
idles at **~1.7 GB GPU**; requests are batch-1 and queue behind a single GPU worker (the offload
residency machinery is deliberately single-flight), so this is an *availability* deployment, not
a throughput one.

**Posture — read before exposing it.** This is a localhost tool for the machine's owner, not a
hardened multi-tenant server. As of 0.6.3 it binds to `127.0.0.1` by default; bind beyond
localhost only on networks you trust (`E4B_HOST=0.0.0.0`), and set `E4B_TOKEN` when you do — then
the generation routes require `Authorization: Bearer <token>` (`/health` stays open for monitors).

```bash
pip install "experts4bit-qlora[serve]"
E4B_ADAPTERS="alpaca=./out/adapter_best.pt" python -m experts4bit_qlora.serve   # 127.0.0.1:8777
# LAN + auth: E4B_HOST=0.0.0.0 E4B_TOKEN=$(openssl rand -hex 16) python -m experts4bit_qlora.serve
```

**Many fine-tunes, one base.** Every adapter in `E4B_ADAPTERS` (plus `base`, the un-tuned model)
is served concurrently over the same NF4 base: adapters live in pinned CPU RAM and hot-swap over
the live LoRA parameters per request (~tens of ms against a multi-second generation), validated
at startup against the model's LoRA key-set — so N fine-tunes cost the VRAM of one. All adapters
must share the server's `R`/`ALPHA` (R is checked structurally; ALPHA is invisible in a `.pt`).

- `POST /generate` — `{prompt, adapter?, max_new_tokens?, temperature?, top_p?,
  repetition_penalty?, stream?, seed?}` → `{text, adapter, tokens, tok_per_s, swap_ms, stopped}`,
  or SSE token events with `stream: true`.
- `GET /health` — status, adapters, queue depth, GPU memory; never blocks behind a generation.
- `POST /v1/completions` + `GET /v1/models` — OpenAI-compatible (`model` selects the adapter).
  Deliberately no `/v1/chat/completions`: OLMoE has no chat template; send Alpaca-format prompts
  (`### Instruction:\n...\n\n### Response:\n`).

Guardrails: `E4B_QUEUE_MAX` waiting requests (then 503 + Retry-After), `E4B_MAX_INPUT_TOKENS`
(413), `E4B_MAX_NEW_TOKENS` clamp, `E4B_REQUEST_TIMEOUT_S` (partial text, `stopped: "timeout"`).
The allocator cache is released to the driver between requests (`E4B_EMPTY_CACHE=1`) so bursty
GPU neighbors can use the headroom.

[`deploy/`](../deploy/) has the Dockerfile + compose file (CUDA 12.4 runtime base, the pinned stack
the A2000 numbers were measured with).

**The Python inside the image must be a real 3.11, not Ubuntu's.** Ubuntu 22.04's
`python3.11` apt package is 3.11.0rc1 (Aug 2022), which predates
`sys.get_int_max_str_digits` — torch 2.6's dynamo polyfills decorate it at import, so the
first `import torch._dynamo` (reached through `transformers.integrations.moe` when loading
any MoE model) half-registers, fails, and every later import dies with the misleading
`Duplicate dispatch rule for <built-in function intern>` while the serve healthcheck keeps
passing. The Dockerfile installs 3.11 from deadsnakes and asserts the fixed interpreter at
build time; if you build your own image, keep both.

One deployment note worth setting: the
container should carry `ulimits: memlock: -1`. On the A2000 stack above, a slow offloaded
decode was seen without it; that card is a correctness-only testbed, so the reading is not
quoted as a speed.

**Correction (2026-07-28): the stated *cause* was wrong.** That note used to say the pinned-RAM
homes "silently fall back to pageable" without the rlimit. They do not — `pin_memory()` /
`cudaHostAlloc` is **not** gated by `RLIMIT_MEMLOCK`. Measured on a RunPod SECURE A6000 whose
memlock was capped at **8 MiB soft and hard**: a **15 GiB** pinned arena allocated fine and moved
at 18.7 GB/s (pinned-class H2D). The rlimit gates `cudaHostRegister` (locking pages you already
own), which this path never calls. The slowdown on that host is **not attributed** — set the
ulimit as cheap insurance, and do not use it to explain a slow path without checking
`tensor.is_pinned()` first.

Bind note (0.6.3+): the compose sets `E4B_HOST=0.0.0.0` **inside** the container (a
container-loopback bind is unreachable through the port map — the container's network namespace
is the isolation boundary) and publishes on the **host loopback** (`127.0.0.1:8777:8777`) by
default. To reach it from the LAN, widen the host publish to `8777:8777` and set `E4B_TOKEN`.

## Continuous-batching server (opt-in, v1)

`experts4bit_qlora.serve_paged` is a second, separate server: an OpenAI-compatible `/v1/completions`
(and `/v1/chat/completions` when the tokenizer has a chat template) over the **continuous-batching
engine** -- `ContinuousScheduler` driving `PagedModelRunner` over the FP8 paged KV -- rather than
over HF `generate`. It exists so that request-level serving benchmarks (TTFT, inter-token latency
and throughput under Poisson arrivals, as `vllm bench serve` and `sglang.bench_serving` drive them)
can be run against e4b exactly as they are run against vLLM, SGLang and llama.cpp. **It is the
server the serving campaign benchmarks against**; it is not the shared-GPU availability deployment
above, and it serves no adapters.

```bash
pip install "experts4bit-qlora[serve,fast]"
E4B_PAGED_MODEL=Qwen/Qwen3-30B-A3B E4B_PAGED_ARENA=/arenas/qwen3-30b.nf4 E4B_PAGED_CALIB=/calib/placement.json \
  python -m experts4bit_qlora.serve_paged          # 127.0.0.1:8778; /health says "loading" until the stack is built
```

The served stack is the harness's stack, built in the harness's order (`bench/p39/step_decomp.py`,
`bench/p44/serve_stack.build_served_model`): NF4 through the arena, placement solved then every expert
in VRAM (`E4B_PAGED_PLACEMENT=all-vram`, the point every certified serving number was measured at),
the hybrid tier, then the int4 levers read from the **same environment names the lane hook uses**
(`E4B_SERVE_EXP_INT4`, `E4B_SERVE_ATTN_INT4`, `E4B_SERVE_ATTN_INT4_CALIB`, ...), amortisation off,
the paged attention, and the fusions at one assembly point as the harness: `fuse_qkv`
(`E4B_PAGED_FUSE_QKV=1`), which applies the env-gated folds (`E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`,
`E4B_FUSE_ROUTER_EPI`) itself -- the registered B=1 fused stack is `--fuse-qkv` with those flags set -- or,
without it, the three folds called directly. A set lever that patches nothing refuses at startup, and
`GET /health` reports the census (int4 expert layers, int4 attention projections, modules each fusion
patched, decode-graph status per bucket) so a reader can tell which stack answered. Its `prefill_routes` block
reports the prefill routes as the forward resolves them, read at each request: `int4_prefill` (`E4B_INT4_PREFILL`
resolved, `auto` -> `k19` or `loop`), `int4_prefill_above_256_rows` (with `device_grouping` on, `k19` only under
`k19`, else `mtile`), `prefill_attn` (`E4B_PAGED_PREFILL_ATTN` resolved), and the raw `*_env` values. A harness
should record that block, not the box's environment.

Those resolved names describe an int4-b32 store and a layer without sinks or a sliding window. They do not say what a
given model ran. On gpt-oss they read `k19` and `flash` while no call takes either: its MXFP4 store serves rows up to
256 through K21 and rows above through the kept NF4 stacks (`E4B_INT4_KEEP_NF4=1`), and every gpt-oss layer has sinks,
so it keeps the explicit mask. **`prefill_routes.seen` is what ran.**
- `seen.moe` counts each expert-GEMM call's route and row class, for example `mxfp4_k21|le256` or
  `nf4_mtile_captured|gt256`.
- `seen.prefill_attn` counts each prefill attention call's path: `flash`, or `explicit_mask:` with `sinks`, `window`
  or `env`.

Both are counted in the Python forward, so eager calls and graph captures count and graph replays do not. A count
says the route ran, not how often a replayed graph did. An engagement check should assert on `seen`.

**First-chunk prefill graph (`E4B_PAGED_PREFILL_GRAPH`, `auto` by default since lane SC2b).** Every first chunk of
exactly `E4B_PAGED_CHUNK_TOKENS` tokens replays one CUDA graph of the prefill forward instead of launching it kernel by
kernel; later chunks, and first chunks of other lengths, run eagerly. A first chunk reads no history, so one graph
serves every slot.

It engages only if it verifies at startup:
- device grouping is on (decode graphs at `max_seqs > 1` on the all-vram placement) and the model has no
  linear-attention state;
- the capture succeeds (a host sync inside the forward fails it);
- on two seeded prompts, each replay equals an eager forward bit for bit, in the logits and in every layer's staged
  K/V.

The three settings:
- **`auto`** (the default) also stands down when the device's free memory after capture is below the graph's private
  pool. The graph keeps its 512-token forward's working set for its life (+3.3 GiB on Qwen3-30B-A3B int4 at 16
  sequences in SC2b), and a later chunk still runs eagerly and needs that working set again. When `auto` stands down,
  the server runs with eager prefill and `/health` says why.
- **`1`** engages, or stops the server at startup with the reason.
- **`0`** prefills eagerly.

`/health`'s `prefill_graph` block reads:
- `status`: `on`, `off`, `refused`, `loading` or `error`;
- `requested`: the setting;
- when engaged, `T`, `replays`, `eager_chunks`, `eager_reasons` (`later_chunk`, `short_chunk`), `pool_mib` and
  `free_after_mib`. The startup check's replays are not counted;
- when refused, `why`.

What SC2b read (#846, `bench/sc2/`, one RTX 5090, Qwen3-30B-A3B int4, 16 sequences, 512-token prompts):
- serial TTFT **1.30-1.65x** faster;
- every request's streamed text byte-identical with the graph off and on, in both draws;
- no regression at any arrival rate;
- the capacity ceiling **unchanged** at 1 req/s. Under load, most of the per-prefill stall on running decodes is work
  outside the graphed forward.

The A2000 census behind it is `bench/prefill-graph-census-2026-10-04/`.

**Bulk KV bookkeeping (`E4B_PAGED_BULK_KV`, on by default since lanes SC2c and SC2d; `0` restores the per-layer path).** Each request costs the engine some KV
bookkeeping outside the model forward:
- a slot reset at admission and at finish;
- the prompt's flush into the FP8 pool once its last chunk runs;
- with decode graphs, a claim of every block the slot can reach, at its first decode.

The per-layer, per-block forms issue about **13.5k launches per request** on Qwen3-30B-A3B at 2048 tokens per slot:
96 + 2,448 kernels + 6,240 copies + 4,608 + 96. They are host-issued one after another on the engine thread, and every
resident decode waits for them. `E4B_PAGED_BULK_KV=1` does the same work in a launch count independent of layers and
blocks:
- `Fp8PagedKV.reset_all_layers`;
- `claim_blocks`: one async table write per request;
- `append_prompt`: one quantize per side and one scatter per region per side, per group of layers of one geometry, a
  group bounded at 16 MiB of input per side.

It leaves the same pool bytes, block tables, lengths and free lists as the per-layer path, with the same rows given to
the same slot. `/health`'s `kv_bookkeeping` block reads `requested` and, per path, how many requests' prompt flushes
(`flush_layers`, `flush_bulk`) and first-decode block claims (`ready_layers`, `ready_bulk`, or `ready_at_flush` when
the bulk flush already made them) the server ran. `flush_bulk_fallback` counts the bulk flushes `append_prompt`
wrote per layer after all (a slot already holding tokens, prompts of different lengths, a demoted arena): the same
bytes, without the saving.

**Memory.** A bulk flush allocates up to `Fp8PagedKV.append_prompt_peak_bytes(T)`:
- its largest layer group's stacks and quantize temporaries, 7× its bf16 input per side, a group bounded at 16 MiB per
  side;
- every layer's FP8 K/V, held until the writes.

That is ~216 MiB on Qwen3-30B-A3B at a 2048-token prompt (~138 MiB at 512). Under an eager forward it reuses memory the forward just
returned. Under the first-chunk prefill graph the forward's working set sits in the graph's private pool, so the flush
is additive: the graph's `auto` headroom check counts the bound at the slot's capacity when bulk is on, and
`/health`'s `prefill_graph.bulk_flush_mib` reports it. `tests/test_bulk_kv.py` compares whole pools, and a tiny model decodes the same tokens either way. The
stall census behind it is `bench/stall-census-2026-10-05/` (exploratory: launch counts and bitwise parity on the A2000,
a correctness testbed, and a post-hoc read of SC2b's traces). On Qwen3-30B-A3B, lane SC2c measured serial TTFT 3.8× faster and
capacity up from 2 to 4 req/s, with identical output ([SC2c](../bench/h2h-2026-10-02/sc2c/README.md)). SC2d confirmed identical
output on Qwen3.6-35B-A3B and gpt-oss-20b ([SC2d](../bench/h2h-2026-10-02/sc2d/README.md)). Neither has a register row yet.

**Per-step trace (`E4B_PAGED_STEP_TRACE=<path>`).** One JSON line per engine step (`engines/step_trace.py`):
- what the step carried: prefill chunks and tokens, prefill-graph replays, decode rows and bucket, slots decoding for
  the first time, admissions, active and queued requests;
- its host time by segment (`ops`, `plan`, `pf_prep`, `pf_forward`, `pf_flush`, `pf_sync`, `pf_emit`, `dec_ready`,
  `dec_prep`, `dec_issue`, `dec_sync`, `dec_mirror`, `dec_emit`, `retire`, `dispatch`), summing to `step_ms`;
- `gpu`: in ms from the step's first event, when the GPU reached `pf_prep` and `dec_prep` and when it finished the
  forward, the flush and the decode (`pf_forward`, `pf_flush`, `dec_issue`). These are read after the step's own
  syncs, so the instrument adds none. The GPU is idle at the two `*_prep` marks, so `pf_forward - pf_prep` is the
  prefill forward's device time.

Its cost is a few `perf_counter` calls and up to six CUDA events a step.

Engine knobs: `E4B_PAGED_MAX_SEQS` (`auto`, below; batch width = KV slots), `E4B_PAGED_MAX_TOKENS_PER_SEQ` (4096;
prompt + output per sequence -- a request past it is a 400, never clamped), `E4B_PAGED_CHUNK_TOKENS`
(512), `E4B_PAGED_MAX_PREFILL_TOKENS` (per-step budget; default = chunk), `E4B_PAGED_GRAPHS` (`auto`, the default:
bucketed CUDA-graph decode on scratch slots on a CUDA device of sm_89 or newer at `all-vram`; `0` eager, `1` forced) +
`E4B_PAGED_BUCKETS` (`1,2,4,8,16`, trimmed to `max_seqs`; `auto` follows `max_seqs`, below), `E4B_PAGED_TRACE=<path>`
(one JSON line per finished request: arrival, admitted_at, first_token_at, finished_at, prompt_len,
out_len, finish_reason -- server-side TTFT/ITL beside the client's), `E4B_PAGED_STEP_TRACE=<path>` and
`E4B_PAGED_BULK_KV` (above), `E4B_HOST` / `E4B_PORT` / `E4B_TOKEN`
as above. `GET /stats` returns the scheduler's `stats()` (TTFT p50/p99 **from arrival**, queue wait,
per-stream rate) and the runner's graph statistics.

**Slots: `E4B_PAGED_MAX_SEQS=auto` by default (lane SC2e, #846; `16` restores the old default).** When the engine
builds, before any weight is read, `auto` takes the widest width lane SC2e read that the serve estimate fits in the
device's free memory (`serve_recipe.choose_max_seqs`):
- **Widths.** 64, 32 or 16 with the default buckets (`auto`, below). 64 or 16 with an explicit list such as
  `1,2,4,8,16`: 32 slots on that list chain every wide step as two 16-row replays, which SC2e did not read.
- **The fit.** `estimate_serve_footprint`'s device total for the width (weights, KV pool and scratch slots, a hybrid's
  per-slot linear-attention state, the bulk KV flush's ceiling), plus a reserve for the first-chunk prefill graph's pool
  (`chunk_tokens x hidden_size x layers x 16 B`: 0.75 GiB on Qwen3-30B-A3B, above the 0.42–0.57 GiB measured), plus
  1.5 GiB, must fit the free memory. On SC2e's box the server used 2.0–2.3 GiB more than the estimate at ready. About
  0.5 GiB of that is the CUDA context, already outside the free memory; the reserve and the margin cover the rest.
- **What it picks for Qwen3-30B-A3B int4.** On an RTX 5090: 64 at 2,048 tokens a slot; 32 at the default 4,096 (64
  needs ~35 GiB). On a 24 GB card: 16. Without a CUDA device, under the solver placement, or when nothing fits: 16, as
  before.
- **What SC2e read** (Qwen3-30B-A3B int4, one RTX 5090, 512-token prompts, 2,048 tokens a slot): 64 slots on the default
  list held the SLO to 8 req/s against 4 at 16 slots, and 64 with `E4B_PAGED_BUCKETS=auto` to 12. Serial TTFT and TPOT
  were within 1 % and serial output byte-identical. Measured on Qwen3-30B-A3B int4 on an RTX 5090; other models get
  the widest width the estimate fits, not separately measured.
- **Outputs under load.** Under load, output text differs from unbatched output: SC2e's 64-slot server produced the
  16-slot server's text on 0.46–0.74 of requests. That comes from batching, not slots, but more slots mean more
  batching.
- `/health` reports `engine.max_seqs` (the width serving), `engine.max_seqs_requested` and
  `engine.max_seqs_resolution` (every candidate's arithmetic and the reason).

**Buckets: `E4B_PAGED_BUCKETS=auto` by default (lanes SC2e and P117, #846; `1,2,4,8,16` restores the old list).**
`auto` captures every power of two below `max_seqs` and then `max_seqs` itself (32 -> `1,2,4,8,16,32`), so the widest
decode step is one graph replay. Up to 16 sequences it reads exactly the old list.
- **Speed.** SC2e's 64-slot server held the SLO to 12 req/s with `auto` and to 8 on the old list, whose 64-row step
  runs as four 16-row replays with a host sync after each (36.5 ms against one 18.4 ms replay).
- **Quality.** P117 ([`bench/p117/RESULTS-p117.md`](../bench/p117/RESULTS-p117.md)) read AT_PARITY. Teacher-forced, one
  64-row piece reads −0.0032 nats against four 16-row pieces (`e4b.serve.p117.wide-bucket-quality.qwen3.5090.2026-10-08`),
  inside the 16-row arithmetic's own neutral perturbations; 32-row pieces and padded 64-row steps pass too.
- **Outputs under load** change more often with wide steps: SC2e's wide-bucket servers produced the 16-slot server's
  text on 0.01–0.12 of requests, at no measured quality cost.
- **Scope.** Measured on Qwen3-30B-A3B int4 on an RTX 5090; other models get the wide buckets on the strength of this
  read, not their own.

**Slots above 16: costs.** An explicit list that stops at 16 runs a decode step over 16 rows as consecutive 16-row
replays and logs that at startup. Costs to weigh before raising either knob:
- **KV pool.** Each slot holds `max_tokens_per_seq` of FP8 KV: 103.5 MiB per slot at 2,048 tokens on Qwen3-30B-A3B
  (`serve_recipe.paged_kv_pool_bytes`), so 16 / 32 / 64 slots hold 1.63 / 3.26 / 6.52 GiB, twice that at 4,096
  tokens. A scratch slot is one KV block, but on a hybrid model it is a full slot of linear-attention state.
- **Routes.** Decode rows above 16 take the paths prefill chunks take today: `Int4Linear` above 16 rows runs cuBLAS on
  its cached bf16 weight, more than 256 routed expert rows (bucket 64 at top-k 8) build the chained tile table, and
  the T=1 folds stop at 64 rows (the server logs a bucket above 64).
- **Graphs.** One more captured graph per bucket; its pool is not priced (`estimate_serve_footprint` says so).

`/health` reports `engine.buckets_requested`, `engine.graph_stats` (per bucket: replays, eager steps, rows, padding
rows) and `levers.kv.pool_mib`, and the step trace counts `dec_pieces` (replays per decode step). Lane SC2e's read is
`bench/h2h-2026-10-02/sc2e/README.md`.

**Decode graphs (#770; lanes P109, P110).** `serve_paged` captures bucketed decode graphs by default
(`E4B_PAGED_GRAPHS=auto`: on a CUDA device at `all-vram`, eager elsewhere; `0` keeps eager decode). This is the path
**every registered serving-speed number** describes: SC1, P96, and P98 to P101.
- **sm_89 or newer.** The graphs need the fused FP8 KV append, whose e4m3 cast Triton compiles only on sm_89+.
  Below that (A100, A6000, RTX 30-series, RTX A2000) `auto` decodes eagerly and the fused append degrades to the eager
  one. `E4B_PAGED_GRAPHS=1` and `E4B_FUSED_KV_APPEND=1` are refused in words.
- **The speed.** P109 ([`bench/p109/RESULTS-p109.md`](../bench/p109/RESULTS-p109.md)) read the default server on one
  RTX 5090, with Qwen3-30B-A3B NF4 at `max_seqs` 16, on an EPYC 7C13 host. Graphs were **×5.60** the eager default
  with 16 concurrent requests (731–748 against 125–131 tok/s) and **×9.02** with one request (99.4 against 10.4–11.0).
  The capture costs +0.055 GiB of peak memory and about 3 s at startup.
- **Why the default waited for P110.** P109 read DIVERGENT:
  - The replay is bit-identical to its own padded eager step.
  - But the graph server's tokens leave the eager default's within 16 tokens on 7 of 16 rows. The cause is the device
    grouping and the bucket padding that graphs bring.
  - Two eager configurations, host and device grouping, diverge as fast without any graphs.
  - Which arithmetic is better is a teacher-forced quality question. P110 answered it.
- **The quality.** P110 ([`bench/p110/RESULTS-p110.md`](../bench/p110/RESULTS-p110.md)) read AT_PARITY. Teacher-forced
  over 48 wikitext windows, the graph arithmetic (device grouping and bucket padding) sits at +0.0004 nats against the
  eager default. That is inside the eager default's own neutral perturbations: a half-batch reads +0.0018, a prefill
  split +0.0005, spreads 0.011–0.013. A halved decode scale reads +1.05. On that, `E4B_PAGED_GRAPHS=auto` became the
  default (#770). Greedy outputs differ from the old eager default's at the bf16 level, at no measured quality cost.
- **One KV-table selection per step** (lane P111, [`bench/p111/RESULTS-p111.md`](../bench/p111/RESULTS-p111.md)). A
  bucket's step selects every layer's block-table and seq-lens rows once, outside the graph, instead of once per layer
  (`E4B_KV_STEP_SELECT`, on by default; `0` keeps the per-layer form). On the same server it decodes identical tokens
  1.036× as fast with 16 concurrent requests and 1.012× with one.
- **Programmatic dependent launch** (lane P113, [`bench/p113/RESULTS-p113.md`](../bench/p113/RESULTS-p113.md)).
  grouped-nf4-gemm's decode-row kernels can launch as programmatic dependents of the kernel before them
  (`GNF4_PDL=1`, sm_90+ NVIDIA). Capped to launches of at most 8 rows (`GNF4_PDL_MAX_ROWS=8`), SC1's int4
  configuration decodes identical tokens 1.0404× as fast with one request and 1.0000× with 16;
  uncapped it costs 16 requests 2.1 %. The capped form is grouped-nf4-gemm's default from
  0.37.0. The default NF4 server reaches only two of the switched kernels, so it was not read there.

**Single-request decode GEMV (grouped-nf4-gemm 0.43.0; lane P116).** On GPUs with at least 160 SMs, decode at
Qwen3-30B-A3B's NF4 expert shapes uses grouped-nf4-gemm's bandwidth-targeted GEMV: 1.24× as fast at one request and
unchanged at 16, with quality within P110's bar (`e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`). It is grouped-nf4-gemm's
default from 0.43.0; `GNF4_GEMV_BW=0` restores dot-pad. Other shapes and smaller GPUs are unchanged.

**Prefill on the int4 expert store (#916; lanes P100, P102).** With `max_seqs` 1 the server leaves
`hot_residency.DEVICE_GROUPING` off. Until P102, every prefill chunk's MoE call on the int4 store therefore ran a
Python loop: one reference decode, a cast, a matmul and a copy per routed expert per projection, paid per chunk per
layer.
- **P100** ([`bench/p100/RESULTS-p100.md`](../bench/p100/RESULTS-p100.md)) measured the loop at 73 % of a 512-token
  chunk. The same configuration's TTFT-4096 ranged 4.8-30.1 s across the four hosts measured, because the loop is
  host-sensitive.
- **`E4B_INT4_PREFILL` picks the route:**
  - `auto` (the default since P102): `k19` where K19 can run, else `loop`;
  - `loop`;
  - `batched`: the same weights decoded in slices, bit-identical;
  - `k19`: K19 at every row count, the loop's operands within one bf16 ulp;
  - `mtile`: the grouped int4 M-tile GEMM, int8 activations.
- **P102** ([`bench/p102/RESULTS-p102.md`](../bench/p102/RESULTS-p102.md), RTX 5090, Qwen3-30B-A3B) read
  `DEFAULT=k19`: TTFT-4096 7.21 s → 1.37 s and TTFT-512 0.85 s → 0.11 s. The prefill-shaped NLL stayed within +0.011 /
  +0.007 ppl of the loop on 12 fresh windows (the calibrated K8 rule).
- With `max_seqs > 1`, prefill rows have always taken the M-tile; `auto` moves them to K19 too.
- **What remained of a k19 prefill:** about 45 % of its device time was prefill attention in fp32 without tensor
  cores. The next paragraph covers it.

**Prefill attention (#960; lane P107).** For a layer without sinks or a sliding window, the paged prefill used to hand
SDPA an explicit boolean lower-right mask with `enable_gqa`. With fewer KV heads than query heads, that combination
rules out the flash and memory-efficient kernels, so SDPA ran its fp32 math backend.
- **`E4B_PAGED_PREFILL_ATTN` picks the route:**
  - `flash` (the default since P107): the same mask as `causal_lower_right(T, t_total)`, which the flash kernel takes in
    bf16. Where no fused kernel applies, SDPA serves the bias itself with the same mask.
  - `math`: the explicit boolean mask (SDPA's fp32 math backend).
- Layers with sinks or a sliding window keep the explicit mask under either value.
- **P107** ([`bench/p107/RESULTS-p107.md`](../bench/p107/RESULTS-p107.md), RTX 5090, Qwen3-30B-A3B) read
  `DEFAULT=flash`:
  - a 4096-token prefill's device time fell from 906 ms to 378 ms;
  - TTFT-4096 fell from 1.98 s to 1.74 s on a CPU-bound EPYC host, where the GPU then idles for most of a prefill;
  - the served-prefill NLL stayed within −0.014 / −0.008 ppl of `math` on 12 fresh windows (the calibrated K8 rule).
- The read covers Qwen3-30B-A3B. Other families with plain full-attention layers are covered by the route's CPU
  equivalence tests (`tests/test_paged_prefill_attn_route.py`), not by a reading.
- **What remains:** host-side launch overhead. About 50,000 device-to-device copies per 4096-token prefill remain under
  both routes.

**Limits, stated:** greedy only (`temperature` must be 0 or absent -- a nonzero value is a 400, not
ignored); no `logprobs`, `echo`, `n > 1`, stop strings or penalties (400s); `ignore_eos`,
`min_tokens`, `stop_token_ids`, `max_tokens` and token-id prompts are honoured. Requests past
capacity wait in the scheduler's FIFO queue (no eviction exists in the engine); `E4B_PAGED_MAX_QUEUE`
can cap in-flight requests with a 503. The module docstring records the engine facts a benchmark
reader needs (EOS handling, the per-sequence window, prefill chunking, graphs). Everything above the
GPU seam is tested on CPU with a fake runner (`tests/test_serve_paged.py`); `build_engine` needs a
CUDA box.

**GPU status.** 0.39.0 shipped this server CPU-tested only. Its first GPU run was lane SC1's proof `sc1a-prove-7`
(experts4bit-qlora#846; 2026-10-02, RTX 5090, Granite-3.1-3B with NF4 experts). `build_engine` built the stack, the lever
census was clean, and at B=1 the decode graph captured and the smoke passed. At B=16 the graph for bucket 2 and every
bucket above it failed to capture, and those buckets ran eagerly. The cause: the server never switched on the batched lane's
sync-free device grouping (`hot_residency.DEVICE_GROUPING`), which `bench/p39/step_decomp.py`'s batched lane sets before
capturing. A T > 1 step therefore took the eager grouping's host sync inside the capture. Fixed on main after 0.39.0
(#874). The fix's GPU run is SC1 proof `sc1a-prove-8` (2026-10-02, RTX 5090, e4b `0a2a0c8`; adertha-receipts `bbfbb31`).
At B=16 all five decode-graph buckets (1/2/4/8/16) captured and the smoke passed. The census reports
`grouping: {device_grouping: true}` at B=16 and the library defaults at B=1. Scope of that evidence: Granite-3.1-3B with
NF4 experts and the unfused fold set. Qwen3-30B-A3B with int4 experts and `fuse_qkv` first runs in SC1's box-A reading.

**Hybrid models (linear attention; read on an RTX 5090 in lanes P97 and P101).** `PagedModelRunner` serves models whose
`config.layer_types` mixes `full_attention` with Gated DeltaNet `linear_attention` layers: Qwen3.5 / Qwen3.6 MoE and
Qwen3-Next.
- **The state.** Each sequence's linear-attention state (the causal-conv window and the recurrent state) lives in a
  per-slot pool (`experts4bit_qlora/engines/linear_state.py`). For each forward, a linear layer receives transformers'
  own `LinearAttentionLayer`, built from the bound rows' pooled state, and its updated state is written back. Only
  attention layers flush K/V to the fp8 pool.
- **What the CPU tests pin** (`tests/test_linear_state.py`): the pool reproduces transformers' DynamicCache to 1e-5
  across chunked prefill and batched decode; and a hybrid model through the runner stays within its all-attention
  control's fp8 error, with greedy tokens equal.
- **Refused:** Mamba-style layers, which transformers also labels `linear_attention` (granite-4.0-h, Nemotron-H, Jamba,
  Bamba); and other state-carrying layer types.
- **`build_engine` for a hybrid checkpoint.** The fp8 KV pool is sized to the attention layers only (paged attention
  maps model layer to pool layer; on Qwen3.6, 10 of 40 layers), and the KV geometry comes from a composite config's
  `text_config`. A CPU test pins the compact pool against a one-layer-per-index pool, bit for bit.
- **On the card (lane P97, [`bench/p97/RESULTS-p97.md`](../bench/p97/RESULTS-p97.md), SUPPORTED).** One RTX 5090,
  through gnf4's fp8 decode kernel, Qwen3.6-35B-A3B, 4 sequences interleaved over 512-token prompts and 255 batched
  decode steps:
  - each sequence's pooled state stays transformers' own: 7.7e-3 relative error at the layers before the first
    attention layer, which see the same tokens on both paths;
  - the whole model tracks transformers' forward at 4.43e-3 nats, argmax agreement 0.972 (OLMoE through the same
    harness: 7.15e-3, 0.974);
  - a slot-mapping mutant reads 4.05 nats.
- **Decode graphs (lane P101, [`bench/p101/RESULTS-p101.md`](../bench/p101/RESULTS-p101.md), SUPPORTED).**
  `E4B_PAGED_GRAPHS=1` serves hybrid models.
  - On one RTX 5090, Qwen3.6-35B-A3B through `build_engine` captured every decode bucket (1, 2, 4, 8, 16) and
    replayed them with no eager step. The graph arm's tokens equal the padded eager step's on all 17 requests.
  - W16 (16 staggered requests) decoded at 449.0 tok/s and W1 (one request) at 79.7, 2.02x and 3.28x plain eager,
    on a Ryzen 9 7950X host. Eager hybrid decode is launch-bound, so its speed, and the ratio, depend on the host
    CPU: P98's EPYC 7663 host ran the same eager arms at about half the speed.
  - The per-slot state is gathered and scattered through the bound bucket's device selector, and the pool is warmed
    and frozen before capture. `tests/test_hybrid_decode_graphs_gpu.py` pins replay against the padded eager step bit
    for bit, on an sm_89+ card.
  - Before the fix, the replays faulted. P98 ([`bench/p98/RESULTS-p98.md`](../bench/p98/RESULTS-p98.md), VOID) hit a
    device-side assert during W16, which P99 ([`bench/p99/RESULTS-p99.md`](../bench/p99/RESULTS-p99.md)) localised to
    bucket 1's first replay.
  - The cause was the MoE engine's single-entry row-to-token index cache: a later bucket's capture warm-up freed the
    index an earlier bucket's graph still read. It is fixed in #918 (#913), with a reproduction test on a non-hybrid NF4 MoE.
- **The Gated DeltaNet kernels: supported and recommended (lane P105,
  [`bench/p105/RESULTS-p105.md`](../bench/p105/RESULTS-p105.md), SUPPORTED; quality and prefill: lane P106,
  [`bench/p106/RESULTS-p106.md`](../bench/p106/RESULTS-p106.md), NEUTRAL).** transformers uses `fla` /
  `causal_conv1d` when they are installed, and its torch path otherwise. For hybrid serving, install the read
  versions:

  ```sh
  pip install flash-linear-attention==0.5.2 causal-conv1d==1.7.0
  ```

  - **Speed.** On one RTX 5090, Qwen3.6-35B-A3B under decode graphs ran 536.5 tok/s against 472.7 on W16 (1.135×), and
    100.7 against 90.1 on W1 (1.118×). That saves 1.6 ms per step at one row and 3.3 ms at sixteen. fla alone carries
    82–83 % of the gain. Every bucket replays exactly as the padded eager step on the kernels.
  - **Correctness.** The paged path's error against transformers running the same kernels stays within 2× an
    all-attention control on every seed of a dense hybrid (`tests/test_linear_state_dense_parity_gpu.py`), on the real
    fp8 kernel.
  - **Quality: no measurable cost (P106).** Both paths ran in one process, compared teacher-forced on wikitext
    (16,384 prompt positions, 512 decode steps) on fp32 log-probs:
    - KL(torch ‖ kernels) 5.7e-3 nats on prompt positions and 4.9e-3 on decode steps, about the fp8 KV's own 4.43e-3
      (P97);
    - argmax agreement 0.969 and 0.973;
    - d_nll +3e-5 and −8e-4 nats.
  - **Prefill (P106).** TTFT at one request runs 1.124× the torch path's at 512 tokens, 1.106× at 2,048 and 1.101× at
    4,096. That is about 32 µs saved per prompt token, roughly 10 % of prefill.
  - **Caveat: token streams differ from the torch path's.** The kernels' arithmetic differs. About 3 % of positions are
    argmax near-ties that flip, so greedy decode diverges from the torch path's after the first one: graph tokens agreed
    with phase t's on 41–57 % of positions. In nats the difference is the one above.
  - **Caveat: chunked prefill depends on chunk boundaries.** fla's chunk kernel moves results by about 3–5e-3 relative
    with where a prompt is split. transformers' own chunked prefill drifts identically.
  - **History.** P103 and P104 ([`bench/p103/RESULTS-p103.md`](../bench/p103/RESULTS-p103.md),
    [`bench/p104/RESULTS-p104.md`](../bench/p104/RESULTS-p104.md)) stopped at their proving rentals on
    `tests/test_linear_state_gpu.py`. That test's single-seed tiny-MoE statistic is bimodal across seeds on every
    kernel set, the torch path included, so its pass is one draw. P105 gates on the dense multi-seed test instead.
- **Not yet done:** only Qwen3.6 has been read on a GPU.
