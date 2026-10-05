### TC1 amendment 46 registered: grouped-nf4-gemm's decoded route against its fused kernels on OLMoE and Qwen3-30B-A3B, the sm_120 gate first (P107-P111; bench and tests only)

- **What it reads.** RD1's licence for the opt-in `GNF4_TRAIN_GEMM=decoded` (grouped-nf4-gemm#487): the full training step on one RTX 5090.
  - Arms: fused vs decoded on the matched arm, two draws a side, venv-e4b, 60 steps.
  - Families: OLMoE (about 64 rows per expert, above RD1's 48-row line) and Qwen3-30B-A3B (about 32, below it).
- **The box's first step: `tc1_decoded_gate`.** It runs grouped-nf4-gemm's compiled tests for the route at the pinned SHA on the box's card.
  - Each run must pass the tests it names, so a SHA without the route cannot pass on `-k` alone.
  - A failure refuses the box before any timing, with exit 19, which names no machine. `decgate.json` is the record.
- **Predictions:**
  - P107: the gate;
  - P108: OLMoE ≤ 0.95, on the median and on every cross-draw ratio (it moves a default);
  - P109: Qwen3-30B-A3B in [0.97, 1.25];
  - P110: held-out within 0.01;
  - P111: peak at most +0.30 GB.
- **The decision.** If P107, P108, P110 and P111 hold, `auto` takes the route on compute capability (12, 0), the card measured, for calls
  with more than 16 groups and at least 48 rows per present group. That is grouped-nf4-gemm's own PR; other 12.x parts stay on today's
  route until measured.
- **Tooling and tests:**
  - harness: `tc1_decoded_gate` and `tc1_decodedab_family`;
  - reducer: the families, the engagement predicate and the P107–P111 scorers, with self-test cases 100–103;
  - tests for the tokens, plus the gate's pass, fail, no-route and no-checkout paths on a stand-in checkout.
