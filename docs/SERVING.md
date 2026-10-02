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
container should carry `ulimits: memlock: -1`, and on the A2000 stack above, omitting it went
with offloaded decode dropping from 1.44 to ~0.4 tok/s.

**Correction (2026-07-28): the stated *cause* was wrong.** That note used to say the pinned-RAM
homes "silently fall back to pageable" without the rlimit. They do not — `pin_memory()` /
`cudaHostAlloc` is **not** gated by `RLIMIT_MEMLOCK`. Measured on a RunPod SECURE A6000 whose
memlock was capped at **8 MiB soft and hard**: a **15 GiB** pinned arena allocated fine and moved
at 18.7 GB/s (pinned-class H2D). The rlimit gates `cudaHostRegister` (locking pages you already
own), which this path never calls. The 3.6× slowdown was real on that host but is **not
attributed** — set the ulimit as cheap insurance, and do not use it to explain a slow path
without checking `tensor.is_pinned()` first.

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
the paged attention, and either `fuse_qkv` (`E4B_PAGED_FUSE_QKV=1`) or the env-gated folds
(`E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`, `E4B_FUSE_ROUTER_EPI`) -- the harness's two exclusive
branches; asking for both is refused. A set lever that patches nothing refuses at startup, and
`GET /health` reports the census (int4 expert layers, int4 attention projections, modules each fusion
patched, decode-graph status per bucket) so a reader can tell which stack answered.

Engine knobs: `E4B_PAGED_MAX_SEQS` (16; batch width = KV slots), `E4B_PAGED_MAX_TOKENS_PER_SEQ` (4096;
prompt + output per sequence -- a request past it is a 400, never clamped), `E4B_PAGED_CHUNK_TOKENS`
(512), `E4B_PAGED_MAX_PREFILL_TOKENS` (per-step budget; default = chunk), `E4B_PAGED_GRAPHS=1` +
`E4B_PAGED_BUCKETS` (`1,2,4,8,16`; bucketed CUDA-graph decode on scratch slots), `E4B_PAGED_TRACE=<path>`
(one JSON line per finished request: arrival, admitted_at, first_token_at, finished_at, prompt_len,
out_len, finish_reason -- server-side TTFT/ITL beside the client's), `E4B_HOST` / `E4B_PORT` / `E4B_TOKEN`
as above. `GET /stats` returns the scheduler's `stats()` (TTFT p50/p99 **from arrival**, queue wait,
per-stream rate) and the runner's graph statistics.

**Limits, stated:** greedy only (`temperature` must be 0 or absent -- a nonzero value is a 400, not
ignored); no `logprobs`, `echo`, `n > 1`, stop strings or penalties (400s); `ignore_eos`,
`min_tokens`, `stop_token_ids`, `max_tokens` and token-id prompts are honoured. Requests past
capacity wait in the scheduler's FIFO queue (no eviction exists in the engine); `E4B_PAGED_MAX_QUEUE`
can cap in-flight requests with a 503. The module docstring records the engine facts a benchmark
reader needs (EOS handling, the per-sequence window, prefill chunking, graphs). Everything above the
GPU seam is tested on CPU with a fake runner (`tests/test_serve_paged.py`); `build_engine` needs a
CUDA box.
