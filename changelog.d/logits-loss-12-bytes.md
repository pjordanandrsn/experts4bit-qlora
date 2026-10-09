### Training estimate: logits and loss at 12 bytes per logit, three fp32 tensors

- **What changed.** `estimate_qlora_footprint`'s activation item prices the loss branch at `T × V × 12` bytes
  (`recipe.LOGITS_LOSS_BYTES`), up from 10 (bf16 + fp32 + fp32). Its detail line now reads "three fp32 logits-sized
  tensors".
- **Why.** An allocator replay of granite-3.1-3b-a800m training on an RTX A2000 found three fp32 logits-sized tensors
  live together at the loss peak, 3 × 192.0 MiB at T = 1024 and V = 49,155 (loggetta#49). OLMoE-1B-7B's
  reference-kernel replay found the same, 3 × 196.5 MiB (loggetta#44). At 10 B per logit the estimate was 2 × T × V
  bytes short wherever the loss is the larger branch.
- **Which receipts it now covers.** In sample: the committed loggetta MoE training receipts, priced with their own
  setup and workload.
  - Now covered (were under the allocated peak): granite-3.1-3b-a800m (−56.0 → +40.0 MiB), granite-4.0-h-tiny
    (−69.0 → +127.0 MiB) and both OLMoE reference-kernel receipts (−9.7 → +88.6 MiB).
  - Unchanged: OLMoE with `grouped_nf4`. There the planner's grouped-kernel backward term is the larger branch.
  - Already over and now more so: Qwen3-30B-A3B on the RTX 5090, +296.8 MiB each.
- **Not modelled.** The chunked LM loss (`enable_fast_train` past 1 GiB of fp32 logits) does not materialize the
  logits whole. This term still prices them whole there.
- **Test.** `tests/test_topology_recipe.py::test_logits_and_loss_are_priced_at_three_fp32_tensors_per_logit`.
