### TC1 amendment 50 registered: bucketed padding as `auto` -- its default decision (P123-P125)

- Token `qwen3fieldauto`: `NF4_QLORA_PAD_BUCKETS=0` vs `auto` (grouped-nf4-gemm#491: buckets only where a call carries >= 16,384 routed
  rows) at TC1's field recipe, whose calls carry at most 9,040. Shipped and matched arms, two load-gated draws a side. P123: `auto` never
  buckets there; P124: held-out within 0.005; P125: matched peak within +0.05 GB. Speed is reported, not scored: both sides run the same ops,
  and amendment 49 showed this baseline's own draws 8-21 % apart. All HELD (with amendment 48's re-ask) makes `auto` grouped-nf4-gemm's default.
- `tc1_arm.py` records the bucket mode grouped-nf4-gemm resolved and its row gate. `tc1_reduce.py`: the family, `fieldauto_why`,
  `score_fieldauto` (structure and values from the VALID receipts, speed reported); self-test 115 cases.
