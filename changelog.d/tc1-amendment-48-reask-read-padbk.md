### Read: TC1 amendment 48's re-ask — bucketed padding takes 4.29 GB off e4b's packed-row peak and steps it 0.893 (P115–P118 HELD)

- `tc1-5090-102` ($1.20, a quiet Xeon W-2145 host): `NF4_QLORA_PAD_BUCKETS=1` (grouped-nf4-gemm#490) against the single padded block on
  packed 4,096-token rows. Matched arm 0.893 [0.892, 0.895] with its peak 32.52 → 28.23 GB; shipped 0.933 [0.932, 0.934]; held-out within
  0.0004. Row `e4b.train.pad-buckets.qwen3.5090.2026-10-06`; STATUS says so. Opt-in until a field-recipe A/B.
