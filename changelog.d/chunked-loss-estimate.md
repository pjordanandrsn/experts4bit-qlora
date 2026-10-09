### Training estimate: the loss branch is priced as the chunked LM loss where the run chunks it

- **What changed.** `estimate_qlora_footprint`'s activation item now prices its loss branch as the run will take it.
  With `expert_kernel="grouped_nf4"`, `enable_fast_train` routes a supported architecture's training loss through the
  chunked LM loss under `E4B_CHUNKED_LM_LOSS`. By default (`auto`) that happens once the stock fp32 logits reach
  `AUTO_MIN_LOGITS_BYTES`, 1 GiB: T ≥ 1,767 at Qwen3's vocabulary, T ≥ 5,462 at granite's. There the branch is now
  `chunked_loss_bytes` (one chunk's logits plus the gathered hidden rows), read from the engine's own table, switch and
  gate. Everywhere else it is unchanged: whole logits at `LOGITS_LOSS_BYTES`.
- **Why.** Measured on an RTX A2000 with granite-3.1-3b-a800m (`grouped_nf4`, resident) at T = 6144 under `auto`: the
  allocated peak is 4105.3 MiB, against 6522.7 MiB with the loss stock. The estimate charged whole logits either way
  and sat 2458.9 MiB over the chunked run. It is now 58.7 MiB over. Stock points are unchanged (+41.5 and +72.5 MiB).
- **Not covered.** Forced chunking at a small T (`E4B_CHUNKED_LM_LOSS=1` at T = 1024, an opt-in setting) is now
  36.2 MiB under. A chunk's own coefficient (`CHUNK_BYTES_PER_LOGIT`, 10) is a stated formula. The allocator replay
  that would attribute it fails inside the chunk's checkpoint recompute, and the coefficient is unchanged here.
- **Evidence.** `bench/chunked-lm-loss/estimate-a2000-granite/`: four receipts and the evaluation. In sample: one
  model, one card.
- **Tests.** `tests/test_topology_recipe.py`: the chunked branch above the gate and the stock branch below it, under
  the reference kernel, with the switch off, with a fixed chunk size, and for an architecture outside the table.
