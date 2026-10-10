# #1469 item 1 — the expert-locality census on Qwen3-30B-A3B (trace only, no routing change)

The census answers how much expert reuse Qwen3-30B-A3B's ordinary router shows across nearby tokens. It reports three
quantities per MoE layer:
- the union of experts in 16-, 32- and 64-token windows;
- top-k churn from token to token;
- the hit rate e4b's hot-residency policy actually gets.

It feeds #1469 items 2 and 3 (the bytes model and the planner decision, owned by `loggetta-e4b-gnf4`), #1470's
speculative-decoding cost question, and the next decode lever (K19 above 256 routed rows).

**Where it runs.** On one rented RTX A4000 (16 GB), a single run under the #846 standing tier (under $15). Expert ids
are correctness-class data, so the A4000 is enough, and no timing from it is quoted.

**Amendment 1 (2026-10-10): RTX A4000, not A2000.** The registration named one rented RTX A2000:
- `loc-a2000-4` and `-5` ran on the 6 GB variant and failed on harness defects, since fixed (#1520, #1527);
- the census then needs about 1.8 GB more GPU memory than calibration used, so the next launch required 12 GB
  (`--vast-min-gpu-ram-gb 12`), and no 12 GB A2000 was offered in 3 h (36 checks).

The RTX A4000 is sm_86 like the A2000, so the census runs the same NF4 host-residency path and kernels. The launch
requires its 16 GB (`--vast-min-gpu-ram-gb 16`). The runner's host-RAM floor is 48 GB, down from 64: the pinned NF4
experts are about 15 GB, and the bf16 checkpoint streams one shard at a time.

The A4000 (sm_86) cannot run the fp8
paged runner, so the trace goes through e4b's NF4 host-residency path: `load_moe_4bit_streaming(offload=True, pin=True, prefetch=True)` and the
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

On the box, `locality_run.sh` runs these four steps after its host refusals, the install (experts4bit-qlora at the
launch commit, grouped-nf4-gemm installed from its clone at `GNF4_SHA`), an import tripwire, the model fetch at `REV`
and both self-tests. The controller starts it through the launcher:

```
GNF4_SHA=<40-hex> bash bench/locality-1469/locality_drive.sh     # the launcher's --command
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

## Results: loc-a4000-1 (2026-10-10), read

| run | card | outcome | cost |
|---|---|---|---|
| `loc-a4000-1` | one rented RTX A4000, 16 GB, sm_86 (Amendment 1) | rc 0, both self-tests 9/9, capture and summaries complete | $0.241 |

The stack was experts4bit-qlora at `d7ba80da` (0.52.0) and grouped-nf4-gemm at `d769d502` (`capture_routing.py` sha256
`d6cc33b2`), with torch 2.8.0+cu128 and transformers 5.16.1, on `Qwen/Qwen3-30B-A3B` @ `ad44e777`. The residency was
the pipelined engine on all 48 layers, holding 8 hot experts per layer from the out-of-sample calibration profile, with
offload, pinning and prefetch. Expert ids are correctness-class data, and no timing from this card is quoted.

The receipts are in `receipts/loc-a4000-1/`:
- `decode.npz` (`expert_ids` int16 `[2048, 48, 8]`: four prompt kinds × 512 greedy steps);
- `prefill.npz` (`[10256, 48, 8]`: 16 teacher-forced wikitext-2 test windows of 641 tokens);
- the four `decode_<kind>.jsonl.gz` (`capture_routing`'s shape) and `manifest.json`;
- `summary_decode.json` and `summary_prefill.json`;
- the calibration profile, plus the box's `summary.txt`, `versions.txt` and `forensics.txt`.

**Distinct experts per layer in a W-token window (of 128).** Each cell is the mean over the 48 layers; the layer range
is in brackets.

| W | decode | prefill |
|---|---|---|
| 1 | 8.0 | 8.0 |
| 16 | 42.4 [32.1–65.1] | 39.0 [26.8–65.5] |
| 32 | 53.5 [40.1–81.7] | 50.8 [34.8–85.5] |
| 64 | 63.9 [47.2–94.3] | 61.6 [43.4–99.7] |
| 128 | 73.0 [54.4–100.1] | 71.9 [53.0–109.1] |

**Churn** (the share of the 8 routed ids that change from one token to the next, mean over layers): decode 0.612
[0.497–0.913], prefill 0.519.

**Hit rate of the hot set (decode).** `1 − cold_gathers / (steps × k)`, from the engine's own counters, is 0.129
[0.083–0.207]. The 8 hot experts of 128 per layer serve about twice the uniform share (6.25 %) of routed slots.

**What it says.**
- At B=1, the routed set moves quickly. A 16-token window already touches about a third of a layer's experts, and a
  128-token window more than half.
- A static hot set of 8 per layer from calibration text catches about one routed slot in eight.
- Items 2 and 3 of #1469 consume these traces. This read registers no prefetch and no residency change.

**Checks against earlier Qwen3 readings (reported, not gated).**
- **S2's** 17- and 33-token windows touched 41.5 and 53.8 distinct experts per layer. Here, decode at W = 16 and 32
  gives 42.4 and 53.5.
- **B1** reported a median Jaccard of 0.33 and a repeat of 0.50 from its H-series traces. The comparison made here is
  the median across layers of the per-layer repeat (1 − churn) and its implied Jaccard r / (2 − r). Prefill gives 0.496
  and 0.330; greedy decode gives 0.396 and 0.247. B1's text has not been traced, so this read does not say which of
  the two is B1's comparison.
- **P100** saw 87 distinct experts per 512-token prefill chunk. W = 512 is not in the registered set; prefill at
  W = 128 gives 71.9.

**Reproduce.** The summaries are recomputed from the committed npz by `tests/test_locality_1469_read.py`, or by hand:

```
python bench/locality-1469/locality_summary.py --npz bench/locality-1469/receipts/loc-a4000-1/decode.npz \
    --manifest bench/locality-1469/receipts/loc-a4000-1/manifest.json --phase decode --out /tmp/summary_decode.json
python bench/locality-1469/locality_summary.py --npz bench/locality-1469/receipts/loc-a4000-1/prefill.npz \
    --phase prefill --out /tmp/summary_prefill.json
```

| file | sha256 |
|---|---|
| `decode.npz` | `f59d7517b7aed61d21f7fc2737ef1db6c6dbbb084122cc5fcf9cc0da801f9148` |
| `prefill.npz` | `17db4e562ff0c7f0303f578d87f90d1dba0c024ebd312de904c8601eb54e8d10` |
| `summary_decode.json` | `f9f0febc519b4e3633e0f3b8648d8769f26954abfaf8c6f6f2e675eea482bd11` |
| `summary_prefill.json` | `11d46d9a296d3b8d79318db09775ba19045aab202085a22311f513f164decd12` |
| `manifest.json` | `16e604262bd6717feb75c77999f8d543468629ed930ffb3758df2f4490859e48` |
