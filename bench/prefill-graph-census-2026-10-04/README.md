# Prefill-graph feasibility census — the NAS RTX A2000, 2026-10-04 ($0)

**Question.** SC2b asked whether `serve_paged`'s prefill forward could be served from a CUDA graph (a proposed
`E4B_PAGED_PREFILL_GRAPH`). Before writing the knob, this census answers two things: what host syncs a 512-token
prefill chunk makes, and whether the forward captures as-is. It is not a registered lane, no claim is drawn from it,
and it measures nothing about speed.

**Answer.**
- The model forward makes **zero host syncs**, on chunk 1 (no history) and chunk 2 (512 tokens of history), on both
  prefill routes.
- A 512-token forward **captures with no code change**, and its replays are **bitwise-equal to eager** on the logits
  and on every layer's staged K/V.
- The blocker for a knob is the Python-side K/V staging (below), not syncs.

## Setup

**Model.** A tiny random Qwen3-MoE (`make_tiny.py`), saved as a local checkpoint with Qwen3-30B-A3B's tokenizer:
- Qwen3-30B-A3B's attention geometry: head_dim 128, 8 query / 2 KV heads;
- hidden 512, 4 layers, 32 experts, top-8, moe_intermediate 256.

It was baked to an NF4 arena by `bench/p39/k8_bake.py`, staged byte-identical (sha256 `e8b8bd21…`). Placement
calibration is `bench/p39/calib.json` (`15b79a10…`), and placement is `all-vram`.

**Engine.** Built by the unmodified `serve_paged.build_engine`. There is one harness shim:
`huggingface_hub.snapshot_download` returns a local directory as itself, so the int4 lever reads the local checkpoint.

**Software.** e4b `6178a15b` (#1063, which adds `/health`'s `prefill_routes`) and grouped-nf4-gemm `71185d6b` (e4b
CI's pin), on torch 2.8.0+cu128, transformers 5.17.0, an RTX A2000 (sm_86) and driver 575.64.05.

**Stack.** SC2's int4 server:
- `E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0`;
- the three folds (`E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1`);
- `E4B_PAGED_FUSE_QKV=1`;
- SC1's `ROUTEENV`.

Both prefill routes were left **unset**, and `max_seqs` 16 and graphs `auto` are the defaults. The server's own
`prefill_routes` read:
- arm A: `int4_prefill` = `int4_prefill_above_256_rows` = `k19`, `prefill_attn` = `flash`, `device_grouping` true;
- arm M (`E4B_INT4_PREFILL=mtile`): `mtile` / `flash` / true.

All four fusions applied (`fuse_qkv_n` 4, `fuse_t1_glue_n` 17, `fuse_t1_glue_r2_n` [4, 4],
`fuse_router_epilogue_n` 4), as did both int4 levers (4 expert layers, 8 attention projections). The decode graphs
fall back to eager: the fp8 paged-attention kernel needs sm_89. Prefill does not use that kernel.

## Instruments and their calibration (`census_prefill_sync.py census`)

1. `torch.cuda.set_sync_debug_mode("warn")`, with each synchronizing op's Python stack. The site is the innermost frame
   outside torch, and the phase is "forward" inside the model's `__call__`, "around" otherwise.
2. `torch.profiler`'s CUDA runtime events: `cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaMemcpy*`.

Run 1 left two gaps, and run 2 closed both. One "around" site in chunk 1 had no frame outside torch; run 2 kept its
whole stack, and it is `set_sync_debug_mode`'s own "prototype feature" notice, not a sync. The profiler also counted
2 `cudaDeviceSynchronize` per chunk; run 2's no-op baseline (`runtime_baseline_noop`) reads the same 2, so they are
the profiler's own plus the census's explicit sync. After that, the two instruments agree exactly:

| chunk | forward syncs | around syncs (site) | runtime stream syncs |
|---|---|---|---|
| c1 (positions 0–511) | 0 | 1 (`paged_runner.py:150`, `ids` from a Python list, a pageable H2D copy) | 1 |
| c2 (512–1023, 512 tokens of history) | 0 | 2 (`:150`, and `:175`, the first token's `int(argmax)`) | 2 |

Arm M is identical. Launches per chunk (4 layers): c1 has 428 `cudaLaunchKernel` + 20 Triton, and c2 has 773 + 20.
c2 also has 1,036 device-to-device copies.

## Capture and replay (`capture`, `replay`)

| test | arm A (k19) | arm M (mtile) |
|---|---|---|
| chunk 1 captures (run 1, `capture`; run 2, `replay`) | yes | yes |
| chunk 1 replays, 3 prompts: logits and staged K/V (4 layers) vs eager | bitwise, max abs 0.0 | bitwise |
| chunk 1 replay after an eager 300-token prefill on another slot + allocator churn | bitwise | bitwise |
| chunk 2 captures over a fixed staged history | yes | yes |
| chunk 2 replays, 2 prompts, vs eager on the same history | bitwise | bitwise |

**Not shown by this run.**
- It did not record that the reference prompts' outputs differ from one another. Distinct tokens give distinct K
  projections, so they do, but that is an argument here, not a measurement.
- The churn probe has no mutation arm, so it is not shown able to fail.

A knob's tests must carry both (see `finding_an_inert_check_disproves_nothing`).

## What a knob must change (design constraints, not syncs)

1. **Staging is Python-side.** `PagedAttentionContext.stage()` appends each chunk's bf16 K/V to a list and `torch.cat`s
   the whole prompt every chunk. Prefill history comes from these bf16 lists, not the FP8 pool, which is written once
   by `flush` when the prompt completes. A replay neither redoes the append nor moves history addresses. The fix is a
   static per-slot staging buffer, written in place. Chunk 1 then needs only the graph's static K/V re-staged after
   each replay.
2. **The same `cat` is c2's 1,036 copies** (4 layers). Its cost grows quadratically in chunks; a static buffer removes
   it with or without graphs.
3. **Inputs.** `ids` and `pos` need static device buffers.
4. **History grows.** History grows by 512 per chunk, so there is one graph per chunk index (8 at 4096 tokens). Each
   graph's pool holds the full-sequence logits: 512 × 151,936 × bf16 = 155 MB on Qwen3. Restricting the head to the
   last row is a separate saving.
5. **Device grouping only.** The knob should require device grouping, which is on by default at `max_seqs > 1`.
   `max_seqs = 1` (host grouping) was not censused.

## Files

- `make_tiny.py`, `census_prefill_sync.py` (run 2's version), `run_census.sh` (the QNAP driver, run 2's arms).
- `run1/`:
  - `census_prefill_sync.run1.py`, the exact script run 1 ran;
  - arms A `census` + `capture` and M `census`.
- `run2/`: arms A `census` + `replay` and M `replay`.
- In both runs, `run.log` is the driver's log, including the pre-flight GPU state. The resident home services held
  7.8 GB (run 1) and 2.1 GB (run 2) of 12 GB, at 0 % utilization.
