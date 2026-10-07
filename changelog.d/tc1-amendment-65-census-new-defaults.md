### TC1 amendment 65 registered: the training-phase census at the new defaults (P184-P188)

- Token `qwen3memc4kr` runs amendment 57's training-phase census on packed rows against e4b's current defaults (the reentrant checkpoint,
  the combine over row chunks, the chunked held-out loss), against e4b with `E4B_CKPT_OFFLOAD=1`, and against Unsloth, one draw each,
  `--mem-census 1`, with no evaluation inside the census window.
- P184: at least 90 % of each peak attributed. P185: e4b defaults 1.3-2.1 GB above Unsloth. P186: with the offload, at most 1.2 GB above.
  P187: every peak in a training step. P188: the offload arm's largest non-static group at the peak is grouped-nf4-gemm's LoRA delta.
- The reducer adds the family, `memc4kr_why` and a generic largest-group prediction in `score_memc4k` (self-test 131).
