### Serve estimate: the cold rows' device stack

- Under the solver, a layer call streams its routed NVMe experts to the GPU and runs them there
  (`hot_residency._cold_contrib`). `estimate_serve_footprint` now prices that stack at its ceiling, `min_hot_rows ×`
  the row bytes, as a device item.
- Measured on an RTX A2000 (OLMoE-1B-7B, solver at 1.2 / 1.5 GiB, a 128-token prompt). Allocator history at the
  generation peak put the whole 183 MiB gap between the estimate and the peak in `_cold_contrib`: 54 routed rows ×
  3.375 MiB plus their outputs.
- At all-VRAM with a short prompt the estimate was 8 MiB over the peak, so nothing there is missing.
- **Prefill staging is priced too.** A prompt's K/V stay bf16, for every attention layer, until the prompt
  completes (paged_attention's staging buffer). The scheduler can hold one finishing prompt plus the next one's
  first chunk.
  - The estimate charges `(max_tokens_per_seq + chunk_tokens) ×` the bf16 K/V bytes per token (at most `max_seqs`
    prompts): its ceiling, since prompt lengths are the caller's.
  - Measured (OLMoE, four 1024-token prompts): 128 MiB staged at the peak, exactly one prompt's worth. That was most
    of the 179 MiB all-VRAM gap; the rest is MoE workspace above the stated working-set heuristic.
