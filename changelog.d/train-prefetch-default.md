### `enable_dense_offload` streams training with `train_prefetch` by default on CUDA (behaviour change; closes the DQ3 → DQ5 line of #1083)

- **What changed.** `train_prefetch` now defaults to `None`, which turns the overlapped training schedule on for every
  CUDA device chain and leaves it off elsewhere. `True` forces it on every chain, as before. `train_prefetch=False` is
  the opt-out: the old synchronous single-slot path.
- **Why.** On an RTX 5090 at 2048 tokens, training with the schedule was bitwise identical to resident on every
  measured link:
  - 1.0023× the resident step time on PCIe gen 5 x16 (DQ3, #1188), against 1.18× synchronous;
  - 1.0050× on gen 4 x16 (DQ5, #1218), against 1.51× synchronous;
  - 2.00× the resident arm's longest trainable sequence (DQ4, #1206).
- **Who is affected.** A model in `train()` mode on CUDA. Inference (`eval()`, including `formats/dense_disk.py`) never
  consults the schedule. The bench arms that compare against the synchronous path (DQ3's S0) pass `False` explicitly.
- **New tests.** Reentrant gradient checkpointing (whose first forward runs under `no_grad` in `train()`): parity, and
  on CUDA the exact per-step prefetch counts. The default on a CUDA chain, the opt-out, and a model split across CPU and
  CUDA, where only the CUDA chain is scheduled and the report shows it.
- **Memory note, documented and not defaulted.** DQ4's streamed arm reached 2.375× instead of 2.00× with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: under the default allocator the longest fitting sequence left
  6.22 GiB reserved but unallocated. That setting is the caller's (it must be set before CUDA initialises), so
  nothing in the engine sets it.
