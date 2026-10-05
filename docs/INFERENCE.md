# Inference — serving the fine-tune you just made

Moved out of the README (2026-08-01). Nothing here is new; the measurements and dates are as
they were.

The adapters were trained against *this exact* NF4 base (same codebook, same per-expert
absmax). `python -m experts4bit_qlora.infer` serves them over that same base — no
re-quantization to GGUF/AWQ, so quantization error at serving time is identical to what
training saw.

```bash
ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer          # generate
OFFLOAD_EXPERTS=1 BENCH_TOKENS=128 python -m experts4bit_qlora.infer     # timed decode bench
```

## What inference mode adds

All `no_grad`-only; training paths are untouched.

- **Decode fast-path** — a single-token forward skips the one-hot expert-mask machinery and
  its per-expert host syncs, looping the token's `top_k` experts with 0-d device indices.
- **Fused 4-bit GEMV** — single-row base projections go through `bnb.matmul_4bit`'s GEMV
  kernel, reading the packed NF4 weight directly instead of materializing the dequantized
  expert. Gated by a per-configuration correctness probe, which passes on **stock
  bitsandbytes 0.49.x**. 4-bit only; 8/16-bit schemes decode via the dequantize path.
- **Prefetched expert offload** (`OFFLOAD_EXPERTS=1`, default `PREFETCH=1`) — layer `L+1`'s
  NF4 experts copy on a side CUDA stream while layer `L` computes. Staging is layer-granular,
  so the schedule is deterministic (no expert prediction) and residency is bounded at two
  layers.

## Measured decode footprint

RTX A2000, OLMoE + the r16 adapter, 128 greedy tokens; big models: base model, 96 tokens.
Full grids in [METHODOLOGY](METHODOLOGY.md) §12.

*These are v0 offload-path figures. The pipelined engine supersedes them for decode — see
[RESIDENCY-ENGINES](RESIDENCY-ENGINES.md) and `bench/RESULTS-informed-hotsets.md`.*

| model | config | peak GPU |
|---|---|:---:|
| OLMoE-1B-7B | resident (experts on GPU) | 4.86 GB |
| OLMoE-1B-7B | offload, serial | 1.45 GB |
| OLMoE-1B-7B | **offload + prefetch** | **1.68 GB** |
| Gemma-4-26B-A4B | resident | OOM |
| Gemma-4-26B-A4B | **offload + prefetch** | **6.16 GB** |
| Qwen3-30B-A3B | resident | OOM |
| Qwen3-30B-A3B | **offload + prefetch** | **4.41 GB** |

Capability, not throughput. The A2000 is the project's correctness-only testbed (a shared
production box), so the decode rates it read for these rows are kept in METHODOLOGY §12 as a
record and are not quoted here as speed: what this table says is what fits, and at what peak.

**Measure your model's speed with the kill-switches, on the card you will run; do not
extrapolate across shapes.**

## Library use

`enable_inference_prefetch(handles)` links the offload handles the loader (or
`offload_model_experts`) returns; `load_moe_4bit_streaming(..., offload=True, prefetch=True)`
does it for you. Serve with the training run's `QUANT_TYPE`.

Kill-switches for A/B: `E4B_DECODE_FASTPATH=0`, `E4B_INFER_GEMV=0`.

## Over HTTP

See [SERVING](SERVING.md) for the FastAPI shim, Docker deployment, endpoints, env knobs, and
the localhost-by-default / `E4B_TOKEN` posture.
