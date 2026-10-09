# #1469 item 1 — the expert-locality census on Qwen3-30B-A3B (trace only, no routing change)

The census answers how much expert reuse Qwen3-30B-A3B's ordinary router shows across nearby tokens. It reports three
quantities per MoE layer:
- the union of experts in 16-, 32- and 64-token windows;
- top-k churn from token to token;
- the hit rate e4b's hot-residency policy actually gets.

It feeds #1469 items 2 and 3 (the bytes model and the planner decision, owned by `loggetta-e4b-gnf4`), #1470's
speculative-decoding cost question, and the next decode lever (K19 above 256 routed rows).

**Where it runs.** On the RTX A2000, under a `gpu:a2000` claim, with no rental. Expert ids are correctness-class data, so
the A2000 is enough, and no timing from it is quoted. The A2000 (sm_86) cannot run the fp8 paged runner, so the trace
goes through e4b's NF4 host-residency path: `load_moe_4bit_streaming(offload=True, pin=True, prefetch=True)` and the
pipelined residency engine.

**What it reuses.** grouped-nf4-gemm's `bench/cold-engine/routing-trace/capture_routing.py`, from a pinned checkout and
unchanged:
- its four prompts, read from its source by `ast`;
- router discovery by probe through its `routed_ids`;
- its rank invariants and its environment fingerprint.

The Qwen3 trace is therefore comparable with the 12 committed OLMoE, Granite and Qwen1.5-MoE traces.

## The run (two processes, as `serve.py` documents the profile → residency flow)

```
locality_capture.py --phase calibrate --revision REV --out-dir OUT --hot-profile OUT/calibration_profile.jsonl
locality_capture.py --phase census --revision REV --trace-dir GNF4/bench/cold-engine/routing-trace --out-dir OUT \
                    --hot-profile OUT/calibration_profile.jsonl            # --steps 512 --prompt-tokens 64 --hot-per-layer 8
locality_summary.py --npz OUT/decode.npz  --manifest OUT/manifest.json --phase decode  --out OUT/summary_decode.json
locality_summary.py --npz OUT/prefill.npz --phase prefill --out OUT/summary_prefill.json
```

1. **Calibration.** The expert profile over 8 wikitext-2-raw-v1 *validation* windows picks the hot set per layer: the
   top 8 by tokens routed, out of sample for the decode prompts.
2. **Decode.** For each of the four prompt kinds, 512 greedy autoregressive steps after a 64-token prompt, as
   `capture_routing` runs them. The pipelined engine is on, with its traffic counted.
3. **Prefill.** 16 teacher-forced forwards over wikitext-2-raw-v1 *test* windows of 641 tokens (P117's join, stride
   3072), every row recorded.

## Outputs (the format agreed with the owner of items 2 and 3)

- `decode.npz` and `prefill.npz`: `expert_ids` int16 `[n_tokens, n_layers, top_k]` in token order, `seq_offsets` int32,
  `labels` and `n_experts`.
- `decode_<prompt>.jsonl`: `capture_routing`'s shape, beside the 12 existing traces.
- `manifest.json`:
  - the model and revision, the versions and the environment fingerprint;
  - the text provenance with token-id hashes, and the geometry;
  - the residency setup and its counters (`cold_pcie_bytes` and `row_bytes` per layer);
  - the exact command.
- `summary_decode.json` and `summary_prefill.json`:
  - per layer and W ∈ {1, 16, 32, 64, 128}, the union per window (mean, p50, p95, max), in non-overlapping windows
    within sequences;
  - the consecutive churn;
  - for decode, the hit rate `1 − cold_gathers / (steps × k)` from the engine's own counters, never re-simulated.

## Checks against earlier Qwen3 readings (reported, not gated)

- B1's B=1 decode Jaccard 0.33 and repeat 0.50 (`bench/hybrid-g9/b1/`).
- S2's 17- and 33-token windows touching 41.5 and 53.8 distinct experts per layer (`bench/hybrid-g9/s2/`).
- P100's 87 distinct experts per 512-token prefill chunk (`bench/p100/`).
