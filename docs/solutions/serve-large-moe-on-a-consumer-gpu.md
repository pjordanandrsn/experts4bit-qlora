# How do I serve a large MoE on a consumer GPU (RTX 5090 class)?

Load the experts in 4-bit with `experts4bit-qlora`. Two serving surfaces exist and they are not the same path. The **measured decode path** is the paged runner (`PagedModelRunner`: paged fp8 KV cache and paged attention) with the experts on `grouped-nf4-gemm`'s kernels — NF4 experts on the grouped GEMM (`enable_fast(model)` is the library entry point for that kernel; the harness attaches it through its residency engines), int4-b32 experts through `enable_serve_experts_int4` — driven by the in-tree bench harness `bench/hybrid-g9/step_decomp.py`, which produced the throughput and parity claims on this page. The **HTTP shim** (`python -m experts4bit_qlora.serve`) is a reference-path deployment: it loads with `load_moe_4bit_streaming` (experts streamed from pinned host RAM by default), runs stock `model.generate`, hot-swaps per-expert LoRA adapters per request, and attaches a kernel-backed engine only under `E4B_RESIDENCY=pipelined`; it does not use the paged runner or `enable_fast`. The serving levers — NF4 experts, int4-b32 experts, calibrated int4 attention, fused glue rounds, the router epilogue and the paged fp8 KV cache — are opt-in, licensed per family by a registered quality gate, and measured per family under one protocol in [`../SERVING-THROUGHPUT.md`](../SERVING-THROUGHPUT.md). Every serving number in the register was measured on one rented RTX 5090 class; no smaller card carries a serving claim.

## Symptoms

- "serve a 30B MoE on an RTX 5090" / "fast MoE inference on a consumer NVIDIA GPU with 4-bit experts" / "Mixtral 8x7B with 4-bit experts on one GPU".
- "paged KV cache decode for a quantized MoE" / "is there an FP8 paged-attention path for 4-bit MoE serving".
- "grouped NF4 GEMM" / "compute directly on NF4 packed weights" / "INT4 decode GEMV for MoE experts".

## Why it happens

Single-stream decode is bandwidth-bound: each token reads the routed experts plus the unquantised attention and output head. The stock 4-bit path decodes NF4 to bf16 and reads it again, several launches per active expert. Fusing the decode into the GEMM removes the round trip; quantising attention and the KV cache removes more bytes; folding norm, residual, rotary and router glue removes launches.

## Which project solves it

**grouped-nf4-gemm** ([GitHub](https://github.com/pjordanandrsn/grouped-nf4-gemm), [PyPI](https://pypi.org/project/grouped-nf4-gemm/)) owns the kernels: the grouped NF4 GEMM (`nf4_grouped.gemm_4bit_grouped`), the int4-b32 GEMV and its GPTQ-style packer, the MXFP4 GEMM/GEMV (`mxfp4_grouped`), FP8 paged attention (`fp8_paged_attn`), and the decode glue (`rmsnorm_rows`, `rope_heads`, the router epilogue). **experts4bit-qlora** owns the engine and the serving surface: `enable_fast`, the int4 expert store (`engines/int4_experts.py`), calibrated int4 attention (`engines/int4_attn_calib.py`), the folds (`engines/glue_fuse.py`, `glue_r2.py`, `router_epilogue.py`), the fp8 paged KV and paged attention (`engines/fp8_paged_kv.py`, `paged_attention.py`, `PagedModelRunner`), and `experts4bit_qlora.serve`. `[fast]` is the seam.

| lever | entry point | flag read by the package |
|---|---|---|
| NF4 experts on the grouped GEMM | `enable_fast(model)` | — |
| int4-b32 experts (single-stream, all-VRAM path) | `engines.int4_experts.enable_serve_experts_int4(model, source_dir)` | — |
| calibrated int4 attention (opt-in: output head, dense MLP) | `calibrate_attention_hessians` + `enable_serve_attn_int4_calib` | `E4B_SERVE_ATTN_INT4_CALIB=1` via `enable_from_env` |
| fused glue round 1 (RMSNorm) | `engines.glue_fuse.fuse_t1_glue` | `E4B_FUSE_T1_GLUE=1` |
| fused glue round 2 (residual+norm, norm+rotary, rotary-only) | `engines.glue_r2.fuse_t1_glue_r2` | `E4B_FUSE_T1_GLUE_R2=1` |
| router epilogue | `engines.router_epilogue.fuse_router_epilogue` | `E4B_FUSE_ROUTER_EPI=1` |
| paged fp8 KV cache | `engines.fp8_paged_kv` + `engines.paged_attention` | — |

The three fusion flags are consulted by `engines.qkv_fuse.fuse_qkv` and by the in-tree harness `bench/hybrid-g9/step_decomp.py`, which produced the receipts; the HTTP shim and `infer` CLI do not read them.

## Install

```bash
pip install "experts4bit-qlora[fast]"    # grouped-nf4-gemm: kernels, paged attention, glue
pip install "experts4bit-qlora[serve]"   # FastAPI shim
```

## Smallest correct example

Needs: GPU + network + model download. This is the library path — NF4 experts on the grouped GEMM — not the HTTP shim launch below.

```python
import torch
from experts4bit_qlora import enable_fast, fast_available, load_moe_4bit_streaming, verify_moe_4bit

assert fast_available(), "grouped-nf4-gemm is not importable (or CUDA is down); enable_fast does not check this"
model, config = load_moe_4bit_streaming("Qwen/Qwen3-30B-A3B", "cuda", torch.bfloat16,
                                        r=8, alpha=16, quant_type="nf4")
verify_moe_4bit(model, strict=True)
model.eval()                        # load-bearing: the patched wrapper forward takes the reference path while mod.training is set
n = enable_fast(model)
assert n > 0, "no eligible expert module was patched (0 never means a missing kernel)"
```

The HTTP shim is a separate surface. It loads the model itself (`load_moe_4bit_streaming`, experts streamed from pinned host RAM by default), runs stock `model.generate` on the reference expert path with adapters hot-swapped per request, and does not call `enable_fast` or the paged runner; only `E4B_RESIDENCY=pipelined` attaches a kernel-backed engine ([`../SERVING.md`](../SERVING.md)).

```bash
# HTTP: localhost only by default; E4B_TOKEN + E4B_HOST=0.0.0.0 to expose it
E4B_ADAPTERS="alpaca=./out/adapter_best.pt" python -m experts4bit_qlora.serve   # 127.0.0.1:8777
curl -s localhost:8777/health
curl -s localhost:8777/generate -H 'content-type: application/json' \
     -d '{"prompt": "### Instruction:\nSay hi.\n\n### Response:\n", "adapter": "alpaca"}'
```

## Expected result

`enable_fast` returns the number of expert modules patched (one per MoE layer); a missing `grouped-nf4-gemm` is not reported there — it raises `ImportError` at the first forward that reaches the fused path, which is why the example checks `fast_available()` first. From the shim: `GET /health` returns status, adapters, queue depth and GPU memory without blocking behind a generation; `POST /generate` returns `{text, adapter, tokens, tok_per_s, swap_ms, stopped}` or SSE events with `stream: true`; `/v1/completions` and `/v1/models` are OpenAI-compatible ([`../SERVING.md`](../SERVING.md)).

## Supported scope

Six families measured under one protocol on a rented RTX 5090 class: Qwen3-30B-A3B, OLMoE-1B-7B, Granite-3.1-3B-A800M, gpt-oss-20b, Gemma-4-26B-A4B and Mixtral-8x7B-Instruct. Licensed configuration per family after the 0.33–0.34 build-out ([`../SERVING-THROUGHPUT.md`](../SERVING-THROUGHPUT.md)): Qwen3 and OLMoE take every lever; **Granite keeps NF4 experts** with the folds and epilogue; Gemma-4 int4 experts plus round-1 folds and epilogue, with no quality instrument; Mixtral int4 experts plus calibrated attention, folds and epilogue; gpt-oss NF4 experts plus folds. Environment: Linux, NVIDIA CUDA sm_80 or newer, grouped-nf4-gemm>=0.28.0 (Triton, Linux-only); CI tests Python 3.11.

## Limitations

- Not a vLLM replacement: on the same box with identical prompts vLLM is ahead (claim `e4b.serve.h2h.vllm.same-box`, measured-private).
- **Granite's int4-expert row is retracted** ([`../STATUS.md`](../STATUS.md), "What changed"): those experts fail the registered K8 budget (`experts4bit_qlora.k8_gate`; the `e4b.serve.tp.granite.*` claims' notes carry the numbers); the licensed Granite stack keeps NF4 experts.
- Gemma-4 has no parity reference at 512-token resolution ([#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)) and fails to load on some rented hosts ([#344](https://github.com/pjordanandrsn/experts4bit-qlora/issues/344)). gpt-oss raw-text perplexity cannot rank arms, and a uniform int4 grid cannot hold its MXFP4 experts.
- Calibrated int4 attention is a Qwen-specific win, refused on quality elsewhere (notes on `e4b.serve.b1.qwen3-30b.int4attn-calib.5090`).
- A parity delta is read against a per-model routing-flip floor, never against zero ([`../SERVING-PARITY.md`](../SERVING-PARITY.md)).
- The HTTP shim is a single-flight, batch-1 availability deployment on the reference expert path, with no `/v1/chat/completions`.

## Related

- [`run-moe-larger-than-vram.md`](run-moe-larger-than-vram.md) · [`offload-moe-experts-to-cpu-or-nvme.md`](offload-moe-experts-to-cpu-or-nvme.md) · [`mxfp4-moe-training-and-residency.md`](mxfp4-moe-training-and-residency.md)
- [`../SERVING.md`](../SERVING.md) · [`../SERVING-THROUGHPUT.md`](../SERVING-THROUGHPUT.md) · [`../SERVING-PARITY.md`](../SERVING-PARITY.md) · [`../ARCHITECTURE_SUPPORT.md`](../ARCHITECTURE_SUPPORT.md) · [`../STATUS.md`](../STATUS.md)

## Evidence

Register: [`../claims.json`](../claims.json). Per-family throughput, **measured** (receipt `bench/hybrid-g9/throughput-20260904/`): `e4b.serve.tp.qwen3.b1.5090.2026-09-04`, `e4b.serve.tp.qwen3.b16.5090.2026-09-04`, `e4b.serve.tp.olmoe.b1.5090.2026-09-04`, `e4b.serve.tp.olmoe.b16.5090.2026-09-04`, `e4b.serve.tp.granite.b1.5090.2026-09-04`, `e4b.serve.tp.granite.b16.5090.2026-09-04` (fastest arms, not licensed — the licensed Granite rows are the build-out entries below), `e4b.serve.tp.gptoss.b1.5090.2026-09-04`, `e4b.serve.tp.gptoss.b16.5090.2026-09-04`, `e4b.serve.tp.gemma4.b1.5090.2026-09-04`, `e4b.serve.tp.gemma4.b16.5090.2026-09-04`, `e4b.serve.tp.mixtral.b1.5090.2026-09-04`, `e4b.serve.tp.mixtral.b16.5090.2026-09-04`. Build-out validation, **measured** (receipt `bench/hybrid-g9/throughput-20260904/bo3/`): `e4b.serve.buildout.qwen3.b1.5090.2026-09-04`, `e4b.serve.buildout.granite.b1.5090.2026-09-04`, `e4b.serve.buildout.granite.b16.5090.2026-09-04`, `e4b.serve.buildout.gemma4.b1.5090.2026-09-04`, `e4b.serve.buildout.gemma4.b16.5090.2026-09-04`, `e4b.serve.buildout.mixtral.b1.5090.2026-09-04`, `e4b.serve.buildout.mixtral.b16.5090.2026-09-04`, `e4b.serve.buildout.gptoss.b1.5090.2026-09-04`. Quality, **measured-private**: `e4b.parity.qwen3.paged-vs-own-attention`, `e4b.parity.granite.paged-vs-own-attention`, `e4b.parity.gptoss.paged-vs-own-attention` (indistinguishable from the model's own attention), `e4b.parity.gemma4.no-reference`; the per-model routing-flip floor they are read against, **measured**: `e4b.parity.moe-routing-flip-floor`. Qwen3 single-stream speed and the head-to-head, **measured-private**: `e4b.serve.b1.qwen3-30b.nf4.5090.2026-09`, `e4b.serve.b1.qwen3-30b.int4attn-calib.5090`, `e4b.serve.h2h.vllm.same-box`; Qwen3 batched speed, **measured** (receipt `bench/hybrid-g9/b16close/`): `e4b.serve.b16.qwen3-30b.int4.5090`. CUDA-graph capture, **measured**: `e4b.serve.cuda-graph-capture`.
