### `E4B_CHUNKED_EVAL_LOSS=1`: a held-out forward computes its loss from the logits in fp32 chunks (opt-in)

- A `torch.no_grad` forward with labels (an evaluation loop's) ran Hugging Face's loss, which upcasts the whole logits to fp32 and takes the
  log-softmax of the copy. At Qwen3's vocabulary and 4,096 tokens, that is 4.64 GiB above the bf16 logits on an RTX A2000. On TC1's packed
  rows it made the held-out evaluation e4b's run peak: 26.88 GB, against a training-phase peak of 26.59, or 25.85 with
  `E4B_CKPT_OFFLOAD=1` (TC1 amendment 58).
- With the variable set, such a forward runs where its fp32 logits would reach 1 GiB (the training gate's value). It runs without labels,
  so the returned logits are the stock forward's bit for bit, and the loss is computed from them 512 tokens at a time
  (`chunked_lm_loss_from_logits`, the router auxiliary loss added as the forward adds it). The loss transient is 0.58 GiB above the
  logits on the A2000, at 2,048 and 4,096 tokens, and the loss was bit-identical to stock in both
  (`bench/chunked-lm-loss/receipts/eval_loss_peak_a2000.json`, allocator bytes; correctness and memory only).
- Below the gate, or for a forward without labels, with `logits_to_keep` or a tuple return, evaluation is the stock forward. It rides on
  the training patch, so `E4B_CHUNKED_LM_LOSS=0` leaves evaluation stock. `CHUNKED_LM_LOSS_STATS` counts `eval_chunked_calls` /
  `eval_stock_calls`, and TC1's `chunked_lm_loss` receipt records them with the variable.
- It stays opt-in until a TC1 box reads its effect on the run peak.
