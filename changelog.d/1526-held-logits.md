### Training estimate: price the logits a caller holds on the stock-loss path (#1526)

- `estimate_qlora_footprint` adds a device item, "logits held by the caller", of `T x V x 2` bytes whenever the run takes
  the stock LM loss (below the chunked-loss gate, e.g. `T < 1,767` at Qwen3's vocabulary). The stock loss returns its bf16
  logits on the model output. A loop that keeps the output across the backward (`out = model(...); out.loss.backward()`)
  holds them through the whole backward. `LOGITS_LOSS_BYTES` was measured on the chunked loss, which returns no logits.
- The term was derived from the record, not fitted to it. On the reduced Qwen3-30B-A3B probe at T = 1,024 with bf16 expert
  adapters (`bench/issue1526/receipts-29370aaa`), the measured loss branch is 2,085.0 MiB. The stock workspace
  (1,780.5 MiB) plus the held logits (296.75 MiB) is 2,077.25 MiB, a residual of 7.75 MiB (0.16 % of the peak).
- A loop that keeps only the loss (HF Trainer's `compute_loss`) does not pay this. There the estimate over-counts by exactly
  this item. The chunked-loss path and the held-out evaluation item are unchanged.
