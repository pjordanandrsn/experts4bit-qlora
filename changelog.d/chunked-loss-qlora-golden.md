### A golden reference for the chunked LM loss through e4b's QLoRA preparation

- `tests/test_chunked_loss_qlora_golden.py` checks the chunked LM loss against the stock loss on a Qwen3-MoE prepared by
  `prepare_qlora_training` on CPU: frozen NF4 experts, attention and expert LoRA in fp32, a masked prompt and right padding, the router
  auxiliary loss, two accumulated micro-batches, an AdamW step and a second step after it. Seeds and tolerances are declared in the
  module before any measurement: loss within 8 fp32 ulps, every trainable gradient within 1e-5 of its scale, and the first update's
  sign and size above a gradient floor. The chunked side materialises no full logits.
- It is the reference the 4090-fit work on TC1's field recipe builds on (TC1 amendment 74's read: the stock full-vocabulary loss
  backward sets the peak once the padded delta is gone).
