### TC1 amendment 55 registered: amendment 47's packed-row memory census at the current defaults (P141-P143)

- Token `qwen3memc4kb` re-reads the census on packed rows at the current defaults, with bucketed padding `auto`. Three arms, each in
  venv-unsloth: e4b defaults, e4b with `E4B_ABSMAX_DQ=1`, and Unsloth.
- P141: at least 90 % of each peak is attributed. P142: e4b's defaults peak 2.0 to 5.0 GB above Unsloth's. P143: the double-quantized
  absmax brings it within 2.5 GB.
- The reducer's census scorer is now registered per family (`MEMC_SPECS`). Amendment 47's reading is unchanged, and its self-test case
  still passes. New: `memc4kb_why` and self-test case 108 (120 cases).
