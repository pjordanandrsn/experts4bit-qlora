### TC1 amendment 49 registered: grouped-nf4-gemm's bucketed padding at the field recipe (P119–P122)

- Token `qwen3fieldbk`: `NF4_QLORA_PAD_BUCKETS=0` vs `=1` at TC1's field recipe, shipped and matched arms, two load-gated draws a side,
  venv-unsloth, e4b's defaults. P119 / P120: no slower than 1.01; P121: the matched peak not above +0.05 GB; P122: held-out within 0.005.
  All HELD (with amendment 48's re-ask) makes buckets grouped-nf4-gemm's default; a field-recipe cost sends it to a size-gated `auto`
  first.
- `tc1_reduce.py`: the family (TC1's no-loop rule as written), `pad_buckets_why` taking its sides and skipping the chunked-loss check here,
  P119–P122 through amendment 41's scorer; self-test 114 cases.
