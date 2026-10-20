### P133 and P134 records (#1313): the bandwidth GEMV's RTX 4090 and Granite/OLMoE reads, re-derived and claimed

Two served reads set grouped-nf4-gemm 0.47.0's bandwidth-GEMV defaults but had no public record and no claim. Each now
has a results page, public receipts under the citation rules (GPU-UUID, host-name and instance files STORE-ONLY by
sha256), and a register row. The verdicts were re-derived from the private receipt store byte for byte, and they
re-derive from the public records (under Python 3.10).
- **P133** (`p133-4090-2`, RTX 4090): Qwen3-30B-A3B one request 1.5145x as fast, sixteen unchanged (0.9998), quality
  within P110's bar. `e4b.serve.p133.gemv-bw.qwen3.4090.2026-10-18`.
- **P134** (`p134-5090-1`, RTX 5090 at a 400 W limit): Granite-3.1-3b-a800m 1.3979x and OLMoE-1B-7B 1.5011x at one
  request, sixteen unchanged (0.9992, 0.9954), quality within the bar. `e4b.serve.p134.gemv-bw.granite.5090.2026-10-18`,
  `e4b.serve.p134.gemv-bw.olmoe.5090.2026-10-18`.

Files: `bench/p133/`, `bench/p134/`, `docs/claims.json`.
