# INTEGRATION-ASSESSMENT — Serving-engine host for the hybrid CPU/GPU MoE tier

**Status: DRAFT — research assessment only. Human decides — do not implement until confirmed.**
**Date: 2026-08-16. Engines assessed: vLLM v0.27.1 (released 2026-08-11), SGLang v0.5.17 (released 2026-08-08). All links and code states verified against `main` / PyPI on 2026-08-16.**

## 0. What we need from a host engine

We (grouped-nf4-gemm: Triton GPU + planned AVX-512 CPU kernels computing directly on packed NF4/MXFP4 bytes; experts4bit-qlora: runtime/loaders/residency) are adding a three-tier MoE execution path:

1. **Hot** experts resident in VRAM (existing Triton kernels).
2. **Warm** experts computed **in place in DRAM** by CPU kernels — no dequant copy, no format conversion.
3. **Cold** experts streamed NVMe→GPU (existing cold-tier wiring).

Non-negotiable: the **one-artifact invariant** — the packed byte format on disk is the format in DRAM is the format the kernels consume, on both devices. The in-repo reference path keeps bit-exact determinism; the serving engine hosts throughput. Evaluation criteria: (a) seam cleanliness for a custom expert backend consuming pre-packed bytes, (b) conversion pressure on the artifact, (c) LoRA + KV/scheduler surfaces, (d) air-gapped deployment weight.

---

## 1. vLLM (v0.27.1, 2026-08-11)

### 1.1 Plugin architecture

vLLM has a formal entry-point plugin system with five documented groups ([plugin docs](https://docs.vllm.ai/en/latest/design/plugin_system/), accessed 2026-08-16): `vllm.general_plugins` (arbitrary registration code — models, quant methods, loaders), `vllm.platform_plugins` (out-of-tree hardware backends; this is how vllm-ascend/vllm-gaudi/vllm-spyre exist as separate packages), `vllm.io_processor_plugins`, `vllm.stat_logger_plugins`, and `vllm.endpoint_plugins` (custom HTTP routes on the OpenAI server, opt-in). A vLLM-blog post (2025-11-20, [Building Clean, Maintainable vLLM Modifications Using the Plugin System](https://vllm.ai/blog/vllm-plugin-system)) documents the sanctioned pattern of shipping even method-level overrides through `general_plugins` instead of forking. [RFC #19161](https://github.com/vllm-project/vllm/issues/19161) ("Enhancing vLLM Plugin Architecture") tracks further expansion.

Beyond entry points, `main` (as of 2026-08-16, post-v0.27.1) has a **`PluggableLayer`** mechanism (`vllm/model_executor/custom_op.py`): layer classes registered with `@PluggableLayer.register` can be **replaced wholesale by an out-of-tree class** via `register_oot` — "supports out-of-tree (OOT) replacement of the entire layer class at instantiation time, allowing customized initialization and submodule composition." Critically, **`RoutedExperts` — the module that owns the expert weight parameters and executes `quant_method.apply()` — is a `PluggableLayer`** (`vllm/model_executor/layers/fused_moe/routed_experts.py`, verified on `main` 2026-08-16).

### 1.2 Quantization plugin seam

Out-of-tree quantization registration exists since [PR #11969](https://github.com/vllm-project/vllm/pull/11969) (merged **2025-01-19**, closing [issue #11926](https://github.com/vllm-project/vllm/issues/11926)): `@register_quantization_config("gnf4")` on a `QuantizationConfig` subclass, callable from a `general_plugins` entry point. `QuantizationConfig.get_quant_method(layer, prefix)` is invoked per layer and, for MoE layers, returns a **`FusedMoEMethodBase`** subclass (`vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` on `main`: abstract `create_weights`, `get_fused_moe_quant_config`, `apply`, plus `apply_monolithic`/`is_monolithic`, `supports_eplb`, `topk_indices_dtype`). The living in-tree example is `tests/quantization/test_register_quantization_config.py` (present on `main`, verified 2026-08-16).

### 1.3 Where expert forwards dispatch

On `main` (2026-08-16): `FusedMoE` (layer.py, now slim) → `create_moe_runner(...)` factory → **`MoERunner`** orchestrates the forward; expert execution flows through `RoutedExperts.quant_method.apply()`. The factory signature explicitly takes `runner_cls: type[MoERunner] | None` ("Custom MoERunner class") and `routed_experts_cls: type[RoutedExperts] | None` ("Custom RoutedExperts class") — the orchestration and the weight-owning module are both injectable. Below the method, the documented [FusedMoE Modular Kernel design](https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel/) splits execution into `FusedMoEPrepareAndFinalize` (quantize+dispatch/combine) and a permute/experts/unpermute component selected by the quant method's `select_gemm_impl()` — a custom experts kernel plugs in by implementing that component. The open expert-offload RFC (below) states the invariant we care about: "All paths go through `runner.forward()` → `quant_method.apply()`", "the cache is a weight provider, not a special forward path."

### 1.4 Pre-packed bytes without conversion — yes, with precedent

- `create_weights()` allocates parameters of **arbitrary dtype/shape** (e.g. `uint8` packed NF4 + absmax/scale side tensors) and attaches a per-parameter `weight_loader`; the `DefaultModelLoader` streams safetensors tensors through those loaders **byte-for-byte** — no conversion occurs if the checkpoint tensor matches the declared packed layout. Every packed in-tree method (AWQ int32, MXFP4 uint8 blocks+scales, NVFP4) already loads pre-packed bytes through exactly this path. `process_weights_after_loading` is the optional repack hook — we make it a no-op.
- For a fully custom artifact (our gnf4 container instead of safetensors): `register_model_loader("<format>")` on a `BaseModelLoader` subclass is a public API — docstring: "When a load format is not supported by vllm, you can register a customized model loader to support it" (`vllm/model_executor/model_loader/__init__.py` on `main`, verified 2026-08-16; [API docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/)) — selected with `--load-format <format>`. The built-in format table keeps churning (`instanttensor`, `modelexpress` are new on `main`) but the registry API is the stable seam. A custom loader controls materialization entirely, so warm-tier experts can stay mmap'd/host-resident with zero format conversion; the only unavoidable copy is H2D for the VRAM-resident hot tier. (Avoid the GGUF/BitsAndBytes loaders — those do convert.)
- Host-resident expert params are a merged first-class feature: `--cpu-offload-params w13_weight w2_weight` ([PR #34535](https://github.com/vllm-project/vllm/pull/34535), merged **2026-02-14**), with pinning opt-out via `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY`.

### 1.5 Can a plugin route some experts to CPU mid-layer?

Yes, at the quant-method seam. `apply()` receives hidden states plus routing (`topk_ids`/weights) per MoE layer and is ordinary Python: a custom `FusedMoEMethodBase` can partition expert IDs, mask the CPU-assigned ones out of the GPU kernel, and run our AVX-512 kernels on DRAM-resident packed bytes concurrently with the GPU stream (this is exactly the mechanism SGLang's merged KTransformers wrapper uses — see 2.3 — and what vLLM's closed monolithic PR [#31938](https://github.com/vllm-project/vllm/pull/31938) did with AVX/AMX kernels + dual-batch-overlap; it was closed 2026-01-23 for being unreviewable, not infeasible). `is_monolithic`/`apply_monolithic` lets a method take over routing+experts wholesale. Caveats to engineer around: CUDA-graph capture of CPU callbacks (piecewise compilation / splitting-ops config or `--enforce-eager` initially) and EP/DP interplay (also unresolved in vLLM's own RFC).

### 1.6 vLLM's own CPU/hybrid MoE offload — state of play

| Item | State | Date |
|---|---|---|
| `--cpu-offload-gb` (generic per-layer swap, not MoE-aware) | shipped, old | since 2024 |
| `--cpu-offload-params` selective by param name (experts stay in DRAM, streamed H2D per forward; 15→31 tok/s single-user Kimi-K2 NVFP4 on GB300) | **merged** [#34535](https://github.com/vllm-project/vllm/pull/34535) | 2026-02-14 |
| Monolithic hybrid (GPU cache + **AVX/AMX CPU compute** + DBO) [RFC #33869](https://github.com/vllm-project/vllm/issues/33869) / [PR #31938](https://github.com/vllm-project/vllm/pull/31938) | **closed** — "too large to review/pass CI" | opened 2026-02-05, PR closed 2026-01-23 |
| [RFC #38256](https://github.com/vllm-project/vllm/issues/38256) "Incremental MoE Expert Offloading — GPU Cache + Async Pipeline": `ExpertWeightProvider` ABC (`prepare(topk_ids) → ExpertWeightResult`), LFRU GPU cache over CPU pinned backing store | **OPEN**, active | opened 2026-03-26, updated 2026-08-05 |
| [PR #37190](https://github.com/vllm-project/vllm/pull/37190) (`--moe-expert-cache-size`, PR 1 of 3, ~980 LOC) | **OPEN**, in review (9 reviews) | updated 2026-08-14 |

Net: vLLM has **no merged CPU-compute expert path** — its roadmap is stream-to-GPU caching. DRAM-in-place CPU compute is exactly the gap our tier fills, and the `ExpertWeightProvider` seam being negotiated is the natural future attachment point for our residency engine (hot/warm/cold is a provider policy). Maturity of what's merged: static offload is fresh (Feb 2026) but in-tree with CI.

### 1.7 LoRA surface

Multi-adapter batching (`max_loras`, `max_lora_rank`), dynamic `/v1/load_lora_adapter` + `/v1/unload_lora_adapter` (gated by `VLLM_ALLOW_RUNTIME_LORA_UPDATING`), pluggable `LoRAResolver`s (filesystem/HF-hub built-in), and — relevant to us — **LoRA on MoE layers including mixed megatron-2D/peft-3D adapter formats** via `--enable-mixed-moe-lora-format`; layout is declared per adapter (`is_3d_lora_weight`) and unverified — misdeclaration silently produces wrong output ([LoRA docs](https://docs.vllm.ai/en/latest/features/lora/), accessed 2026-08-16). Fits our stamped-adapter flow with a validation step on our side.

### 1.8 KV / paged-attention surface

- **FP8 KV**: `--kv-cache-dtype fp8_e4m3|fp8_e5m2`, optional calibrated scales via llm-compressor, per-attention-head scales on Flash Attention; FA3 computes attention in the FP8 domain. No FP4 KV ([quantized KV docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/), accessed 2026-08-16).
- **External observation/extension is a sanctioned API**: `KVConnectorBase_V1` splits scheduler-side (advertise external KV, build metadata) and worker-side (async load/save around forward, per-layer `save_kv_layer`) roles, with hooks for block-pool binding, required KV layout, stats, cache reset, and **KV events** (`--kv-events-config`) for external index maintenance; LMCache and Dynamo attach here ([API](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/), accessed 2026-08-16). [RFC #44223](https://github.com/vllm-project/vllm/issues/44223) (semantic KV reuse) shows the surface still growing.

### 1.9 Scheduler extension

`--scheduler-cls` accepts any class implementing `SchedulerInterface` (pluggable scheduler merged [PR #14466](https://github.com/vllm-project/vllm/pull/14466), **2025-03-12**; used by vllm-spyre). Documented caveat: the scheduler interface is **not public/stable** — pin versions if we touch it.

### 1.10 Air-gap weight

PyPI `vllm==0.27.1` (checked 2026-08-16): **97 `requires_dist` entries, 73 unconditional**; single `cp38-abi3` manylinux wheel **312.9 MB** (x86_64) bundling all CUDA kernels; pins `torch==2.13.0`. GPU (CUDA) is the PyPI default; **prebuilt CPU x86 wheels exist since v0.17.0** (ships AVX512+AMX and AVX2 `.so`, selected at import; [CPU install docs](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/), accessed 2026-08-16) — from-source CPU build needs GCC ≥ 12.3. FlashInfer is **optional**. Guided-decoding deps (xgrammar, outlines_core, llguidance, lm-format-enforcer) are prebuilt wheels. Mirror burden: one fat wheel + torch stack; estimated venv ~7–9 GB (estimate; wheel/dep counts above are the hard numbers). Some noise deps (opentelemetry ×4, `mcp`, `anthropic`, opencv-headless) but all wheel-installable offline. Minimum inference-only install = `pip install vllm` from a local mirror; no runtime downloads besides the model.

---

## 2. SGLang (v0.5.17, 2026-08-08)

### 2.1 MoE backend abstraction

`python/sglang/srt/layers/moe/` on `main` (verified 2026-08-16): a `moe_runner/` package with per-kernel backends (`triton`, `deep_gemm`, `flashinfer_{trtllm,cutlass,cutedsl}`, `marlin`, `aiter`, `ascend`, `humming`, …), `token_dispatcher/`, `topk.py`, plus `kt_ep_wrapper.py` (below). The method seam mirrors vLLM's ancestry: `sglang.srt.layers.quantization.base_config.FusedMoEMethodBase` with `create_weights`, `create_moe_runner`, `apply` (file header: "Adapted from … vllm v0.5.5"). **No entry-point plugin loader exists anywhere in the repo** (code search for entry-point plugin registration, 2026-08-16: zero hits) — extension means an in-tree PR, a fork, or import-time patching of registry dicts.

### 2.2 Quantization surface

`QUANTIZATION_METHODS` is a hard-coded in-tree dict in `sglang/srt/layers/quantization/__init__.py` (verified on `main` 2026-08-16); there is **no out-of-tree registration API** (vLLM's `register_quantization_config` was not carried over). Adding `gnf4` = PR or monkeypatch.

### 2.3 The CPU/GPU hybrid MoE story (the KTransformers integration)

- **[Issue #11425](https://github.com/sgl-project/sglang/issues/11425)** "[Feature] KTransformers Integration to Support CPU/GPU Hybrid Inference for MoE Models" — opened **2025-10-10** by the KTransformers (SOSP'25) team, assignee Atream. This is the solicitation/umbrella issue. Status: **CLOSED as stale** (bot label "inactive", closed **2026-02-08**) with roadmap items 2–6, 8 unfinished (hybrid quant config; GPTQ/AWQ weight formats; `experts_map` instead of contiguous `num_gpu_experts`; hotness-aware placement; **unit tests**).
- Merged implementation PRs: [#11487](https://github.com/sgl-project/sglang/pull/11487) "init support for KTransformers Heterogeneous Computing" (**2025-10-21**), [#12536](https://github.com/sgl-project/sglang/pull/12536) int8 fix (2025-11-03), [#12586](https://github.com/sgl-project/sglang/pull/12586) expert deferral (2025-11-05), [#12834](https://github.com/sgl-project/sglang/pull/12834) refactor to wrap any GPU quant backend (2025-11-10), [#13983](https://github.com/sgl-project/sglang/pull/13983) Qwen3-VL (2025-11-26). Announcement: [LMSYS blog, 2025-10-22](https://www.lmsys.org/blog/2025-10-22-KTransformers/) (up to 20× prefill / 4× decode single-GPU; 227.85 tok/s DeepSeek-R1 on 8×L20 + Xeon).
- Mechanism (on `main` today, `kt_ep_wrapper.py`): `KTEPWrapperMethod(FusedMoEMethodBase)` **wraps any GPU quant method**; experts with ID ≥ `kt_num_gpu_experts` are masked to `-1` for the GPU kernel and computed by `kt_kernel.KTMoEWrapper` (AMX-INT4/INT8/AVX-512) on CPU, with per-token expert deferral. Server flags on `main`: `--kt-weight-path`, `--kt-method`, `--kt-cpuinfer`, `--kt-threadpool-count`, `--kt-num-gpu-experts`, `--kt-max-deferred-experts-per-token` (`server_args.py`, verified 2026-08-16).
- **Conversion pressure — the load-bearing defect for us**: the CPU tier requires a **separate, pre-converted CPU weight artifact** (`--kt-weight-path` pointing at AMX-format weights generated offline; GPU weights load separately). Two artifacts per model, exactly the invariant we refuse. Hot/cold split is static by expert index, not learned placement (roadmap item never done).
- Current health: open PR [#20516](https://github.com/sgl-project/sglang/pull/20516) "Fix KTransformers MoE compatibility regressions in the current SGLang" (updated **2026-08-04**) and open WIP [#34673](https://github.com/sgl-project/sglang/pull/34673) restoring KT offload for DeepSeek-V4 on Ascend (2026-08-13) — i.e., the merged hybrid path **bitrots** because it has no CI (roadmap item 8 unfinished). Related open: [#20126](https://github.com/sgl-project/sglang/pull/20126) UVM-based expert offloading, all-GPU compute (2026-04-27), unmerged.

Net: SGLang is the only engine that **shipped** a CPU/GPU hybrid expert tier, and its wrapper-around-any-quant-method design is the right shape — but the umbrella issue is closed-stale, the CPU tier is a second artifact in a foreign format, and the path demonstrably regresses without dedicated maintainers.

### 2.4 LoRA surface

`max-loras-per-batch` (default 8), S-LoRA/Punica-style batched kernels with default **csgmv** backend (20–80% latency win claimed), dynamic `/load_lora_adapter` / `/unload_lora_adapter`, adapter GPU-pinning, `--enable-lora-overlap-loading` (~35% median TTFT reduction); still listed "under development": embedding-layer LoRA, unified paging, cutlass backend ([docs](https://docs.sglang.io/advanced_features/lora.html), accessed 2026-08-16). LoRA on MoE expert layers is not advertised.

### 2.5 KV surface

- **HiCache**: multi-tier KV (GPU L1 / host L2 / storage L3) extending RadixAttention, with a **pluggable storage-backend seam** (file, Mooncake, 3FS, NIXL) — the strongest KV-extension story of either engine ([LMSYS blog, 2025-09-10](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/); [design doc](https://docs.sglang.io/advanced_features/hicache_design.html)).
- KV events exist (`kv_events_config` in `server_args.py` on `main`, verified 2026-08-16).
- KV dtypes: `fp8_e5m2`/`fp8_e4m3` **and `fp4_e2m1`** (`server_args.py` + `fp4_kv_cache_quant_method.py`/`kvfp4_tensor.py` on `main`, 2026-08-16) — ahead of vLLM on FP4 KV.

### 2.6 Scheduler extension

`--schedule-policy` (lpm/fcfs/…) and `--schedule-conservativeness` tune the built-in scheduler; there is **no `--scheduler-cls`-style class injection** and no plugin hook (verified via `server_args.py`, 2026-08-16).

### 2.7 Air-gap weight

PyPI `sglang==0.5.17` (checked 2026-08-16): **128 `requires_dist` entries, 73 unconditional** — and the GPU kernel stack is now core (no `[srt]` extra): `sglang-kernel==0.4.5` (x86 wheel of the current line: **377 MB**), `flashinfer_python[cu13]==0.6.15.post1` (**mandatory**), `flash-attn-4>=4.0.0b18`, `sgl-deep-gemm==0.1.5.post1`, `tilelang==0.1.11`, `helion==1.4`, `humming-kernels[cu13]==0.1.10`, `nvidia-cutlass-dsl[cu13]==4.6.0`, `quack-kernels>=0.6.1`, `tokenspeed_mla==0.1.8`, `torchao==0.17.0`, pins `torch==2.11.0` and `transformers==5.12.1` exactly, floor `cuda-python>=13.0`. Per-python wheels (cp310–cp313), main wheel 22.3 MB. Air-gap traps: **FlashInfer downloads cubins at runtime** unless `flashinfer-cubin` + `flashinfer-jit-cache` are pre-installed, and the jit-cache wheels are hosted on `flashinfer.ai`'s own index, not PyPI ([FlashInfer install docs](https://docs.flashinfer.ai/installation.html), accessed 2026-08-16); deep-gemm/tilelang/helion JIT-compile on first run, so offline boxes need warmed kernel caches baked into the image. ~9 separate exactly-pinned GPU-kernel packages to mirror per release; estimated venv ~9–12 GB (estimate). A Xeon-AMX full-CPU backend exists (whole model on CPU) but is not the hybrid tier.

---

## 3. Comparison table

| Criterion | vLLM v0.27.1 | SGLang v0.5.17 |
|---|---|---|
| Out-of-tree plugin loading | Entry-point groups (5) + `PluggableLayer.register_oot`; no fork needed (docs + `main`, 2026-08-16) | None; PR, fork, or monkeypatch (code search 2026-08-16) |
| Custom quant method out-of-tree | `register_quantization_config` (merged 2025-01-19) | Not available; in-tree dict only |
| MoE expert-backend seam | `FusedMoEMethodBase.apply` + injectable `RoutedExperts`/`MoERunner` classes + modular-kernel `select_gemm_impl`; `ExpertWeightProvider` RFC open (2026-08-05) | `FusedMoEMethodBase.apply` + `moe_runner` backends; `KTEPWrapperMethod` precedent wraps any method |
| Pre-packed custom bytes, no conversion | Yes: `create_weights` arbitrary dtype + per-param `weight_loader` + `register_model_loader` custom `--load-format` | Yes mechanically (same `create_weights` lineage), but no loader/format registry; KT path ships a **second converted artifact** |
| Mid-layer CPU expert routing | Feasible at quant-method seam (precedent: closed PR #31938); nothing merged | **Merged & shipped** (KT, Oct-Nov 2025) but stale issue, regression PR open (2026-08-04), separate CPU weights |
| Built-in expert offload today | `--cpu-offload-params` static DRAM-resident streaming (merged 2026-02-14); dynamic GPU cache in review (#37190, 2026-08-14) | KT hybrid CPU-compute (merged 2025-10/11, bitrotting); UVM offload PR open |
| LoRA | Multi-adapter, dynamic load/unload, resolver plugins, **MoE LoRA incl. mixed 2D/3D** | Multi-adapter (csgmv), dynamic load/unload, pinning; embedding/unified-paging WIP; no MoE-expert LoRA |
| KV | FP8 KV (e4m3/e5m2, calibrated); `KVConnectorBase_V1` + KV events — external KV mgmt is a public API | FP8 **and FP4** KV; HiCache 3-tier with pluggable storage backends; KV events |
| Scheduler | `--scheduler-cls` injection (merged 2025-03-12; "not public" caveat) | Policy knobs only, no injection |
| Air-gap | 97 deps (73 uncond.), one 313 MB abi3 wheel + torch; CPU wheels since v0.17.0; FlashInfer optional | 128 deps (73 uncond.), ~9 pinned kernel packages (~1 GB+ of wheels) + mandatory FlashInfer with off-PyPI cubin/jit-cache; CUDA-13 floor; JIT warm-up needed |
| License | Apache-2.0 | Apache-2.0 |

---

## 4. Third options

None beats both on the plugin-seam criterion, so none is recommended. **TGI** is eliminated outright: maintenance mode as of 2025-12-11 and the repo archived read-only on 2026-03-21 ([HF docs](https://huggingface.co/docs/text-generation-inference/index); [repo](https://github.com/huggingface/text-generation-inference)). **TensorRT-LLM** requires engine compilation — the maximal violation of the one-artifact invariant — and has no seam for foreign packed formats or CPU expert compute; **lmdeploy** similarly has no out-of-tree backend surface. **llama.cpp server** is the closest prior art for the tier itself — `--cpu-moe`/`--n-cpu-moe` override the buffer type of routed-expert tensors so they live and compute in DRAM while attention/router/shared experts stay on GPU — but it is GGUF-only (our artifact would need conversion), has no plugin mechanism for foreign byte formats, and its LoRA/KV/scheduler surfaces are far below either candidate. KTransformers standalone has become a kernel library (`kt-kernel`) whose serving surface *is* SGLang. Treat llama.cpp and kt-kernel as design references (buffer-type override; AMX kernel scheduling), not hosts.

## 5. Recommendation

**Recommendation: vLLM hosts the throughput/serving surface, as an out-of-tree plugin package (working name `e4b-vllm`); SGLang is the runner-up. Human decides — do not implement until confirmed.**

Rationale against the four criteria:

- **(a) Seam.** vLLM is the only engine where our entire backend ships without a fork: `general_plugins` entry point → `register_quantization_config("gnf4")` → custom `FusedMoEMethodBase` (hot/warm/cold dispatch in `apply()`), plus `RoutedExperts` being OOT-replaceable via `PluggableLayer.register_oot`, plus `register_model_loader` for the artifact. SGLang's seam is the same class shape but reachable only by PR/fork/monkeypatch, and its shipped hybrid (KT) is currently regressing with the umbrella issue closed-stale.
- **(b) One-artifact invariant.** vLLM's loader/quant seams let packed NF4/MXFP4 bytes flow disk→DRAM→kernel unconverted, with `--cpu-offload-params` as merged precedent for DRAM-resident expert params. SGLang's only hybrid path institutionalizes a second converted artifact (`--kt-weight-path`), i.e., the engine's existing convention points away from our invariant.
- **(c) LoRA + KV.** vLLM: MoE-layer LoRA with dynamic adapters (matches the stamped-adapter flow) and a public external-KV API (`KVConnectorBase_V1` + KV events) + `--scheduler-cls`. SGLang counters with HiCache and FP4 KV — genuinely better KV tiering — but that is not the surface our tier extends.
- **(d) Air-gap.** One abi3 wheel + torch vs. ~9 exactly-pinned kernel packages, mandatory FlashInfer with off-PyPI cubin wheels, and runtime-JIT warm-up. vLLM is materially cheaper to mirror and reproduce offline.

**Watch items / risks:** (1) vLLM's [RFC #38256](https://github.com/vllm-project/vllm/issues/38256)/[PR #37190](https://github.com/vllm-project/vllm/pull/37190) may land an `ExpertWeightProvider` API — track it and adopt it as our residency engine's attachment point when merged; until then target `FusedMoEMethodBase` (stable shape since early 2025) and pin the vLLM version in the plugin. (2) The scheduler interface is explicitly non-public — avoid depending on it. (3) CUDA-graph capture around CPU callbacks needs the splitting-ops/eager escape hatch first, graphs later. (4) The `RoutedExperts`/`MoERunner` refactor is newer than v0.27.1 — the plugin should carry a small version-shim layer, per the vLLM plugin blog pattern (2025-11-20).

**Runner-up and switching cost.** SGLang, chosen if vLLM review culture blocks the provider RFC direction or the plugin surfaces regress. Switching cost is bounded and mostly glue, by design: the kernels (grouped-nf4-gemm) and the residency engine (experts4bit-qlora) are engine-agnostic; SGLang's `FusedMoEMethodBase` is literally adapted from vLLM's, so the expert-backend core ports nearly 1:1, and `KTEPWrapperMethod` is a working in-tree template for the wrapper shape. What does not port: entry-point packaging (SGLang has none — becomes a fork or in-tree PR, with the CI-ownership burden the KT integration shows is real), the loader hook (SGLang has no load-format registry), and the LoRA/KV glue. Estimate: the engine-adapter layer is a rewrite measured in weeks; the months-scale work (kernels, residency, artifact) is unaffected. The in-repo bit-exact reference path is the hedge either way — the serving engine is always replaceable because it is never the source of truth.
