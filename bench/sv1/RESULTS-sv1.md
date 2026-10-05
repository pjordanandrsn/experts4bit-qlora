# SV1 read — `sv1-5090-1`

Registered in `bench/sv1/SV1-PREREG.md` (#1153) before the box; work item #1152.

**The run.**
- One RTX 5090: Vast instance 54316730, driver 595.84, AMD EPYC 7B13.
- **$1.496**; teardown proven (destroy HTTP 200, instance absent from the list after).
- experts4bit-qlora `547599b` (#1153's head, 0.48.0) with grouped-nf4-gemm 0.41.0 (`dc8f94a`), torch 2.8.0+cu128.
- Arenas baked on the box: OLMoE in 47 s, Qwen3-30B-A3B (16.3 GB) in 260 s.
- Every arm: 16 requests of 1,024 seeded prompt tokens, 32 new tokens each.
- The lane finished on the box (`TC1_SUCCESS`, five receipts). The launcher's status is HARNESS_ERROR because the
  lane owner stopped the final fetch: it was copying the 16 GB arena the runner left behind (#1160). The receipts
  here were saved from the box first.

Receipts: `receipts/sv1-5090-1/*.json` (`sv1-arm/1`), `summary.txt`, `forensics.txt`.

| arm | decode graphs | prefill graph | estimate | allocator peak | reserved peak | driver peak | tok/s |
|---|---|---|---|---|---|---|---|
| olmoe_eager | off | off | 9.196 | 8.812 | 8.898 | 9.506 | 33.9 |
| olmoe_graphs | on (buckets 1–16 captured) | off | 9.213 | 8.870 | 9.123 | 9.811 | 143.3 |
| olmoe_prefill | on | on, 16 replays, pool 224 MiB | 9.213 | 9.104 | 9.514 | 10.207 | 148.8 |
| qwen3_graphs | on (buckets 1–16 captured) | off | 21.786 | 21.612 | 21.895 | 22.619 | 47.7 |
| qwen3_prefill | on | on, 16 replays, pool 310 MiB | 21.786 | 22.169 | 22.637 | 23.375 | 49.8 |

GiB unless marked. "Estimate" is `estimate_serve_footprint`'s device total for the arm's `ServeSetup`. It excludes the
CUDA context and allocator slack, so it is read against the allocator peak.

**Readings.**
- **S1 (estimate, eager): HELD.** The OLMoE eager peak is 4.2% under the estimate, inside the registered ±5%.
  - Nearly all of the gap is the prefill-staging item, priced at its ceiling (4,096 + 512 tokens). These prompts were
    1,024 tokens. Re-priced at the staged length, the estimate is within ~20 MiB of the peak.
- **S2 (decode graphs, OLMoE):** +60 MiB allocated, +230 MiB reserved over the eager arm. Decode throughput rose 4.2×.
- **S3 (prefill graph):**
  - OLMoE: +240 MiB allocated over the graphs arm (the runner's own pool: 224 MiB).
  - Qwen3-30B-A3B: +571 MiB allocated (pool: 310 MiB).
  - At NF4 the graph costs 0.2–0.6 GiB. SC2b's +3.3 GiB was the int4 stack.
- **S4 (30B):** `qwen3_graphs` peaked 0.8% under the estimate (21.612 against 21.786).
- **Integrity: clean.** Every arm finished all 16 requests. Every captured bucket reported `graph`. Both forced prefill
  graphs reported `status: on` with 16 replays.

**Also recorded.** The CUDA context, driver peak minus reserved peak, was 0.61–0.74 GiB. Allocator slack was
1.0–4.5%; it was largest with the prefill graph's pool.
