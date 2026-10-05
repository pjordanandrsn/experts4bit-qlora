### Read: TC1 amendment 36 — the compact padded LoRA delta is about 3 % faster and the matched peak rose 0.23 GB (P69, P70, P71 FALSIFIED; P72 HELD); it stays opt-in

- `tc1-5090-80` ($1.48, EPYC 7B13, 60-step load-gated draws, venv-unsloth): `NF4_QLORA_COMPACT_DELTA` 0 vs 1. Matched 0.969
  [0.962, 0.977], peak 27.490 → 27.719 GB; shipped 0.970 [0.957, 0.983], peak unchanged; held-out within 0.001.
- The registered reason was memory; the box found speed instead. By the rule it stays opt-in. The single node's backward keeps the padded
  output gradient live while it rebuilds the input block, which is where the extra peak points; releasing intermediates at their last use
  is grouped-nf4-gemm's own change, and a speed default would be its own registration.
