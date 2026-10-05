### serve_paged: opt-in bulk KV bookkeeping (`E4B_PAGED_BULK_KV`) and a per-step trace (`E4B_PAGED_STEP_TRACE`)

- **Why.** SC2b left most of the per-prefill stall under load outside the graphed forward. Each request also costs
  `serve_paged` its KV bookkeeping: on Qwen3-30B-A3B at 2048 tokens per slot, about 13.5k host-issued launches. They are
  the slot resets, the prompt's flush into the FP8 pool, and, with decode graphs, a claim of every reachable block at
  the slot's first decode. All are serialized on the engine thread ahead of every resident decode. The stall census
  (`bench/stall-census-2026-10-05`, exploratory) counted them; its A2000 runs are correctness and counts only (the
  testbed policy), and the time they cost on a 5090 box is SC2c's to measure.
- **`E4B_PAGED_BULK_KV=1`** does that work in a launch count independent of layers and blocks:
  `Fp8PagedKV.reset_all_layers`, `claim_blocks` (one async table write per request) and `append_prompt` (one quantize
  per side and one scatter per region per side, per byte-bounded group of layers of one geometry). It leaves the pool
  bytes, block tables, lengths and free lists exactly as the per-layer path does, with the same rows for the same slot
  (`tests/test_bulk_kv.py`, whole-pool comparisons; a tiny model decodes the same tokens either way). **Off by default**:
  no request-level effect is claimed until a registered lane reads one.
- **Memory, stated.** A bulk flush allocates up to `Fp8PagedKV.append_prompt_peak_bytes(T)` (~216 MiB on Qwen3-30B-A3B
  at 2048 tokens). Under the prefill graph that is additive to the graph's private pool, so the graph's `auto` headroom
  check counts it when bulk is on, and `/health` reports `prefill_graph.bulk_flush_mib`. The bound is checked against
  the allocator's measured peak on CUDA.
- **`E4B_PAGED_STEP_TRACE=<path>`**: one JSON line per engine step. It records what the step carried, its host time by
  segment, and when the GPU finished the forward, the flush and the decode, read after the step's own syncs. `/health`
  reports `engine.bulk_kv` and `step_trace_path`.
