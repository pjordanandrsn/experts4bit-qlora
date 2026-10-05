### Read: SV1 (#1152) — the serve estimate held beside decode graphs and the prefill graph (bench and receipts only)

- `sv1-5090-1` on one RTX 5090, $1.50. All five arms finished with integrity clean: `serve_paged` all-VRAM, 16 ×
  1,024-token prompts. Results in `bench/sv1/RESULTS-sv1.md`; receipts in `bench/sv1/receipts/sv1-5090-1/`.
- **S1 HELD:** the OLMoE eager peak is 4.2% under the estimate. That margin is prefill staging priced at its 4,096-token
  ceiling; at the staged length the estimate is within ~10 MiB.
- **S4:** Qwen3-30B-A3B with decode graphs peaked 0.8% under the estimate.
- **The two unpriced pools, measured at NF4:**
  - decode graphs: +60 MiB allocated (OLMoE; decode throughput 4.2×);
  - the first-chunk prefill graph: +240 MiB (OLMoE) and +571 MiB (Qwen3-30B-A3B), against the runner's own pool of
    224 and 310 MiB. SC2b's +3.3 GiB was the int4 stack.
