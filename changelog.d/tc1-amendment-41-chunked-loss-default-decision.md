### TC1 amendment 41 registered: e4b's chunked LM loss at the field recipe, its default decision (bench and tests only)

- **Why.** Amendment 39 found e4b out of memory on packed 4,096-token rows at the full-vocabulary fp32 logits; #1142's opt-in
  `E4B_CHUNKED_LM_LOSS` never materialises them, and amendment 40 reads whether it fits that regime. A default must also cost the field
  recipe's short rows nothing it should not.
- **The box** (token `qwen3chunkab`): `E4B_CHUNKED_LM_LOSS` 0 vs 1 on the shipped and the matched arm, venv-unsloth, 60 steps, load-gated
  draws. One-sided: P90 / P91 speed ≤ 1.01, P92 matched peak ≤ +0.05 GB, P93 held-out within 0.005. With amendment 40's P89, all
  HELD makes it e4b's default in `enable_fast_train`.
- `tc1_run.sh` gains `tc1_chunkab_family`; the reducer reads it with amendment 36's scorer, whose side names now come from
  `COMPACT_SPECS` (amendments 36-38 re-reduce to identical verdicts), and checks engagement on the `chunked_lm_loss` record (one new
  self-test case).
