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

[`deploy/`](https://github.com/pjordanandrsn/experts4bit-qlora/tree/v0.6.2/deploy/) has the Dockerfile + compose file (CUDA 12.4 runtime base, the pinned stack
the A2000 numbers were measured with). One deployment note that costs 3.6× if missed: the
container needs `ulimits: memlock: -1` — without it the pinned-RAM homes silently fall back to
pageable and offloaded decode drops from 1.44 to ~0.4 tok/s.

Bind note (0.6.3+): the compose sets `E4B_HOST=0.0.0.0` **inside** the container (a
container-loopback bind is unreachable through the port map — the container's network namespace
is the isolation boundary) and publishes on the **host loopback** (`127.0.0.1:8777:8777`) by
default. To reach it from the LAN, widen the host publish to `8777:8777` and set `E4B_TOKEN`.
