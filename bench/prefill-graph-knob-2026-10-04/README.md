# `E4B_PAGED_PREFILL_GRAPH` verified on the NAS RTX A2000, 2026-10-04 ($0)

The knob's GPU verification before merge: its GPU tests, the neighbouring suites, and the knob end to end through the
unmodified `serve_paged.build_engine` on two tiny random MoE models. This is not a registered lane, no claim is drawn
from it, and it measures nothing about speed (that is lane SC2b's job).

**Environment.** torch 2.8.0+cu128, transformers 5.17.0, grouped-nf4-gemm v0.38.0 (`5a887c48`), RTX A2000 (sm_86),
driver 575.64.05. The harness is the census's (`../prefill-graph-census-2026-10-04/`) plus a `knob` mode, staged
here as `census_prefill_sync.py`. The driver is `run_knob.sh`.

## Run 2, at `45d91b3b` (the fix): every arm passes

| arm | result |
|---|---|
| T1: `tests/test_prefill_graph_gpu.py -v` | 7 passed: both mutation arms (no copy; positions dropped after capture) and every refusal fired |
| T2: neighbours (`test_prefill_graph`, `test_serve_paged`, `test_rt_cache_graph_gpu`, `test_decode_graph_buckets`, `test_kv_step_select`, `test_scheduler`) | 82 passed; 5 skipped (sm_89+ decode kernels) |
| K1: tiny Qwen3-MoE, int4 experts + int4 attention + the folds + `FUSE_QKV` (SC2's int4 stack), knob on | engaged; 5 prompts (512, 512, 1024, 512, 300 tokens), each prefilled through the graph on one slot and eagerly on another: first token and every pool layer's K/V **bitwise**; counters 4 replays, 2 eager (1 `later_chunk`, 1 `short_chunk`) |
| K0: as K1 at `E4B_PAGED_MAX_SEQS=1` | refused at startup: `E4B_PAGED_PREFILL_GRAPH=1 refused: device grouping is off: ...` |
| KG: tiny GraniteMoE, NF4 store, SC2's Granite env (`GR_ENV`), knob on | engaged; the same 5-prompt comparison **bitwise**; the same counters |

**The NF4 store verifies.** A Granite server with the NF4 store engages the knob, and its startup check passes.
These are tiny random models, so a real checkpoint is verified again by the same startup check on every engaged
server.

## Run 1, at `9818c557` (the first version): the main bitwise test FAILED

- **Failures.** `test_replayed_first_chunks_prefill_exactly_as_eager` failed: the first tokens of D and B differed
  from eager. K1 engaged, then compared its pool against eager: bitwise for prompt 1 only, and not for prompts 2–4.
- **Cause.** The graph kept its input ids but not its positions tensor, a local of `enable_prefill_graph`. The startup
  check ran in the same scope while the positions were alive, so it passed. Once the method returned, the block was
  freed and reused, so every later replay read garbage positions.
- **Diagnosis.** `dbg_replay_vs_eager.py` replayed after `enable_prefill_graph` had returned. Every replay differed
  from eager by a constant per prompt: logits max abs 0.887 (A), 0.915 (B) and 0.922 (D). The eager forward stayed
  stable.
- **Fix (`45d91b3b`).**
  - The capture returns everything it reads that was allocated outside it, and the graph keeps those tensors.
  - The startup check now runs only after the capture's scope has returned and the allocator has been churned with
    a sentinel.
  - A new mutation test drops the positions after capture, and the check must refuse it (it does: run 2, T1).
- **Harness bug.** KG aborted (device-side assert): the harness drew token ids up to 100,000 and Granite's vocabulary
  is 49,155. Run 2 draws them inside the vocabulary.
- **Script.** `census_prefill_sync.run1.py` is the exact harness run 1 ran.

The census's own replays (`../prefill-graph-census-2026-10-04/`) kept every input alive in one scope, so they could
not see this class of bug.
