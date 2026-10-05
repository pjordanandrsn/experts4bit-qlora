### TC1 amendment 40 registered: the packed 4,096-token regime again, e4b with its chunked LM loss (bench and tests only)

- **Why.** Amendment 39 found every e4b arm out of memory at step 1 on packed 4,096-token rows (the fp32 full-vocabulary logits, 2.32 GiB)
  where Unsloth trained at 24.86 GB. #1142's opt-in chunked LM loss never materialises those logits.
- **The box** (token `qwen3samestack4kce`): amendment 39's box with `E4B_CHUNKED_LM_LOSS=1` on every e4b arm, 40 steps. P87 Unsloth/e4b
  in [0.80, 1.60]; P88 the environment in [0.80, 1.00]; P89 every e4b arm completes resident. A stable reading is recorded, labelled
  opt-in, whichever side it favours.
- `tc1_arm.py` records a `chunked_lm_loss` block on every e4b arm (requested, has it, chunked and stock calls, run-time fallbacks); the
  reducer requires it on this family (`chunked_lm_loss_why`) and reads the box with amendment 39's scorer (one new self-test case).
