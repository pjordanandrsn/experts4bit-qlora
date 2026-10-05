# The prefill graph's `auto` default, verified on the NAS RTX A2000, 2026-10-04 ($0)

The A2000 verification for making `E4B_PAGED_PREFILL_GRAPH` default to `auto`, the default lane SC2b licensed (#846).
This is not a lane. It draws no claim and measures nothing about speed.

**Environment.** grouped-nf4-gemm v0.39.0 (`a5edec87`), torch 2.8.0+cu128, transformers 5.17.0, RTX A2000. The
harness is the census's, staged here as `census_prefill_sync.py`, with its `knob` mode; the driver is
`run_knob_auto.sh`. The models are tiny random ones, as in `../prefill-graph-knob-2026-10-04/`.

## Run 2, at `41ec649f`: every arm passes

| arm | setting | result |
|---|---|---|
| T1 | `tests/test_prefill_graph_gpu.py -v` | 8 passed, including the headroom refusal |
| T2 | neighbours | 85 passed, 5 skipped (sm_89+ decode kernels) |
| KA | **nothing set** (int4 Qwen3-MoE, SC2's stack) | `auto` engaged: pool **188 MiB**, 4805 MiB free; 5 prompts bitwise against eager in first token and pool; counters 4 replays, 2 eager |
| KA0 | nothing set, `E4B_PAGED_MAX_SEQS=1` | the server is built and serves eagerly; `prefill_graph` reads `refused`, `why` = device grouping is off |
| K10 | `E4B_PAGED_PREFILL_GRAPH=1`, `MAX_SEQS=1` | the server stops at startup: `E4B_PAGED_PREFILL_GRAPH=1 refused: device grouping is off …` |
| KGA | nothing set (Granite NF4 store, `GR_ENV`) | `auto` engaged: pool 128 MiB; 5 prompts bitwise |

## Run 1, at `13c90134`: two faults, both fixed before the PR

1. **The pool measure undercounted.**
   - Run 1 measured the graph's private pool as the growth of `torch.cuda.memory_reserved` across the capture. That
     read **16 MiB** for the same graph run 2 measures at 188 MiB, and **0** in the full GPU-test file, once an
     earlier graph's freed segments were recycled.
   - The headroom rule compares free memory against that number, so the undercount made it fail open:
     `test_auto_stands_down_when_the_pool_leaves_too_little_memory` "DID NOT RAISE" in the full-file run.
   - **Fix:** the pool is now the sum of the allocator snapshot's segments whose `segment_pool_id` is the graph's
     `pool()`. A pool that measures 0 is a failed measurement (every capture owns at least one segment), and `auto`
     refuses it rather than engaging unchecked.
2. **A test fixture bug.** `monkeypatch.undo()` also undid the `grouping` fixture's device grouping. The test now
   restores only `mem_get_info`.

The KA0 arm also failed in run 1, inside the harness: it bound slot 8 on a one-slot server. The server itself had
been built and had refused correctly. Run 2's harness records a refused `auto` and returns.

## Files

- `run1-13c90134/` and `run2-41ec649f/`: each arm's JSON and log, plus `run.log`.
