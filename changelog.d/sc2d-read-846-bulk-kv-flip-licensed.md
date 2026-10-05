### SC2d read (#846): bulk KV bookkeeping engages and changes nothing on Qwen3.6 (hybrid) and gpt-oss; FLIP_LICENSED (bench only)

- **The reading** (`sc2d-5090-4`, $0.739; lane total $0.887 over 5 receipts, three of them host or launcher refusals
  before any workload). Both models were served OFF then ON through `serve_paged`'s HTTP server; it was Qwen3.6's
  first run there. Both models read **ENGAGED_IDENTICAL**:
  - gpt-oss-20b: sinks on 24 of 24 layers.
  - Qwen3.6-35B-A3B: the pool holds its 10 attention layers out of 40.
  - On every ON server the bulk path *wrote* every prompt (`flush_bulk_fallback` 0, e4b#1174), with the block claims
    made at the flush. Output was byte-identical OFF against ON in both models, and OFF was deterministic.
- **FLIP_LICENSED**, as registered. With SC2c's licence (#1166) the `E4B_PAGED_BULK_KV` default flip may now cite
  both reads, in its own PR.
- **Correctness only.** The recorded TTFT and TPOT are not read as speed.
