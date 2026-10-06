### TC1 amendment 48 registered: grouped-nf4-gemm's bucketed LoRA-delta padding on packed rows (P115–P118)

- Token `qwen3padbk`: `NF4_QLORA_PAD_BUCKETS=0` vs `=1` (grouped-nf4-gemm#490) on amendment 39's packed 4,096-token rows, the shipped and
  matched arms, two load-gated draws a side, venv-unsloth, e4b's defaults. P115: the matched peak falls by at least 3.0 GB; P116 / P117:
  no slower than 1.02; P118: held-out within 0.005. All HELD leads to a size-gated `auto` and its field-recipe A/B, not straight to a
  default.
- `tc1_reduce.py`: the family (packed predicates, the loop as a recorded route), `pad_buckets_why` (the delta body from grouped-nf4-gemm's
  per-path counters), P115–P118 through amendment 41's scorer; self-test 113 cases.
