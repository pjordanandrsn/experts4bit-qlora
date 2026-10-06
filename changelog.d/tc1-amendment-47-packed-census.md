### TC1 amendment 47 registered: the memory census on packed 4,096-token rows (P112–P114)

- Token `qwen3memc4k`: amendment 23's census (`--mem-census 1`) on amendment 39's packed rows, one draw per arm in venv-unsloth. The arms are
  e4b's library defaults; e4b with `E4B_ABSMAX_DQ=1` + `NF4_QLORA_COMPACT_DELTA=1`; and Unsloth. P112: at least 90 % of each peak
  attributed. P113: e4b's defaults 5–10 GB above Unsloth (amendment 43 read 7.6). P114: with both levers, at most 3 GB above.
- `tc1_reduce.py`: the family (packed predicates, the loop as a recorded route), `memc4k_why`, `score_memc4k`; self-test 112 cases.
