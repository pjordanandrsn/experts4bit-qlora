### The chunked LM loss's workspace is priced at the measured 12 bytes per chunk logit

- **What changed.** `CHUNK_BYTES_PER_LOGIT` was a stated 10 (bf16 logits + fp32 upcast + fp32 gradient). It now equals
  `LOGITS_LOSS_BYTES`, 12, defined once in `engines/chunked_lm_loss.py`, which `recipe` imports. This raises
  `chunked_loss_bytes` by `2 × chunk × V`.
- **Why.** Peak allocated around `chunked_causal_lm_loss`, measured with no recorder (it cannot run inside the chunk's
  checkpoint recompute, #1504): 12.0, 12.06 and 12.07 B per chunk logit at T = 1024, 2048 and 4096 (512-token chunks,
  V = 49,155, H = 1536, RTX A2000). A chunk holds three fp32 logits-sized tensors, as whole logits do.
- **Effect.**
  - **MoE training estimate:** the four granite points are unchanged. Where the chunk term applies, the `grouped_nf4`
    backward branch exceeds it.
  - **loggetta's dense estimate:** it reads `chunked_loss_bytes`. Its chunked-loss plans rise by 178.1 MiB at Qwen3's
    vocabulary. But three Qwen3-32B plans at 2,048 tokens fall by 185.1 MiB: the dense activation formula compares
    unscaled layer work with the loss workspace, so a larger loss term can switch it to a smaller branch. Merging waits
    on that formula's fix in loggetta.
- **Test.** `tests/test_topology_recipe.py::test_whole_and_chunked_logits_are_priced_at_one_measured_coefficient`.
