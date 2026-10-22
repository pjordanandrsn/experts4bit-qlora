### The plan and the chunked loss read one vocabulary width

- `describe_moe` now takes `vocab_size` from the meta tree's LM head (`lm_head.out_features`). That is the width the chunked LM loss's
  run-time gate tests (`labels.numel() x lm_head.out_features x 4` against 1 GiB). The planner's loss terms and the run therefore pick the
  same loss branch. Without an output head, the config's `vocab_size` is still used. The provenance records which source was read.
- `prepare_qlora_training` refuses, by name (`ValueError`), a loaded model whose LM head is not the width the setup was priced at
  (`recipe.vocab_width_refusal`, for example a head resized after the topology was described). It refuses before gradient checkpointing,
  LoRA or any engine is enabled, so a mismatch is never picked silently near the gate.
- `tests/test_vocab_width_one_source.py` covers the width source, the refusal of a `resize_token_embeddings` head, and the refusal in
  `prepare_qlora_training` before anything is enabled.
