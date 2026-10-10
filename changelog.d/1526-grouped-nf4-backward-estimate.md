### The training estimate prices the grouped_nf4 MoE backward (#1526): estimates rise for grouped_nf4 callers

`estimate_qlora_footprint` now prices the working set of one MoE layer's backward when experts train through the grouped
kernel (`expert_kernel="grouped_nf4"`). That working set is:
- the padded LoRA delta, bounded by grouped-nf4-gemm's bucket and pad rules (mirrored here, pinned against the
  installed package);
- the fused workspaces;
- one layer's recompute set.

It is a third branch of the `activations` item's `max()`, beside the loss and the two-layer recompute. Before this, the
estimate left that working set to callers. On a reduced Qwen3-30B-A3B at 4,096 packed tokens with fp32 adapters, the
estimate alone was 19% short of the allocated peak (RTX A2000, `bench/issue1526`).

- **Estimates rise** for every `grouped_nf4` setup where the backward branch is the largest: most at long micro-batches
  and with fp32 adapters. The `reference` kernel is unchanged.
- **In sample:** the four probe receipts in `bench/issue1526/receipts` are the points the branch was set against.
  `tests/test_grouped_nf4_backward_estimate.py` pins estimate >= measured on them.
- **Callers that priced this themselves:** Loggetta 0.5.0 adds its own line as the excess over the `activations` item.
  Against this estimate that excess is zero, so nothing is counted twice. Its tests expect the line and need its
  follow-up.
