### TC1 amendment 38 registered: the compact padded LoRA delta's default decision, a third host and a second family (bench and tests only)

- **Why.** Two hosts read the compact delta on Qwen3-30B-A3B at 0.967–0.970 (matched) and 0.948–0.970 (shipped); with
  grouped-nf4-gemm#473 the matched peak fell 0.288 GB. Amendment 37's two-sided band kept it opt-in on the favourable side, and a
  grouped-nf4-gemm default changes every family, with no family but Qwen3 read.
- **The box** (tokens `qwen3compactab3`, `mixtralcompactab`): amendment 36's A/B on a third host, then Mixtral-8x7B's matched arm
  resident at defaults. One-sided predictions: Qwen3 speed ≤ 0.99 on both arms (P77, P78), its matched peak ≤ +0.05 GB (P79); Mixtral
  ≤ 1.01 (P80), its peak ≤ +0.05 GB (P81); held-out within 0.005 (P82, P83). All HELD makes it grouped-nf4-gemm's default.
- The reducer reads both with amendment 36's scorer (`COMPACT_SPECS`); one new self-test case.
