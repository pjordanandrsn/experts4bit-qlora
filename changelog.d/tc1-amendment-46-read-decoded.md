### Read: TC1 amendment 46 — grouped-nf4-gemm's decoded route saves no step time on OLMoE (1.005) and costs Qwen3-30B-A3B 6.6 % (P108 FALSIFIED); `auto` stays as it is

- **The box:** `tc1dec-5090-4`, $1.07, an AMD EPYC 7713 host.
- **The gate passed first.** grouped-nf4-gemm's compiled tests for the route ran on the 5090 before any arm, and passed (P107).
- **The step:**
  - OLMoE decoded/fused is **1.005** [0.979, 1.031], against a registered ≤ 0.95 (P108 FALSIFIED);
  - Qwen3-30B-A3B is **1.066** [1.050, 1.081] (P109 HELD);
  - held-out and peak are unchanged on both (P110, P111).
- **What it means.** RD1's per-call win does not reach the training step, so grouped-nf4-gemm's `auto` is unchanged and
  `GNF4_TRAIN_GEMM=decoded` stays opt-in.
- **Records:** rows `e4b.train.decoded-route.{olmoe,qwen3}.5090.2026-10-06`, the README section, the STATUS sentence and the receipts.
