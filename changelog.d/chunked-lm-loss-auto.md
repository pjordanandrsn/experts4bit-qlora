### Training: `E4B_CHUNKED_LM_LOSS=auto` -- chunk the loss only where the fp32 logits are large

- **Why.** The chunked loss is what lets e4b train packed 4,096-token Qwen3 rows (TC1 amendments 39, 40), and at the field recipe it
  is a cost: 1.049 of the shipped arm's step on a host-bound RTX 5090 (TC1 amendment 41, P91 FALSIFIED), where a micro-batch's fp32
  logits are 0.3-0.6 GiB. One switch for both regimes needs to tell them apart.
- **What.** `auto` chunks a training forward (512-token chunks) only when its stock fp32 logits -- positions x vocabulary x 4 bytes,
  read from the labels' shape before anything runs -- would reach `AUTO_MIN_LOGITS_BYTES` (1 GiB); a smaller forward runs the stock
  forward untouched and is counted in `CHUNKED_LM_LOSS_STATS["small_calls"]`. The gate sits between TC1's field recipe (0.64 GiB at
  most over a 60-step run, 0.86 GiB for the two longest of its 1,200 rows) and one packed 4,096-token row (2.32 GiB). Mixtral's
  32,000-token vocabulary stays under it at 4,096 tokens (0.49 GiB). `enable_chunked_lm_loss(..., min_logits_bytes=)` is the direct
  call; `enable_fast_train` and the CLI trainer read it from the environment. Opt-in, like `1`; the default is unchanged.
- **Tests.** `tests/test_chunked_lm_loss.py`: a forward one byte under the gate is the stock forward exactly (loss and every gradient
  `torch.equal`, logits returned), one at it chunks; re-enabling moves the gate; the gate separates the TC1 shapes it was set
  between; `enable_fast_train` carries it from `E4B_CHUNKED_LM_LOSS=auto`. The TC1 arm's receipt block records `small_calls` and
  `patched`.
