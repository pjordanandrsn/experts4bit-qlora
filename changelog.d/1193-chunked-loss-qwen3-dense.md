### `chunked_lm_loss` supports dense `Qwen3ForCausalLM` (#1193)

- **The change.** Dense Qwen3's post-`lm_head` code in transformers 5.18.0 is the identity with no aux loss, so it
  joins the chunked-loss table as an `_identity` row.
- **The tests.** `test_loss_and_every_gradient_match_stock[qwen3]` pins it against the stock loss and gradients. The
  refusal test now uses `LlamaForCausalLM`.
- **Opt-in as before.** Needed by lane DQ4 (#1083), whose capacity pairs must not be bound by the full-vocabulary
  logits.
