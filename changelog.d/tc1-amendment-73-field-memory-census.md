### TC1 amendment 73 registered: what holds the field recipe's memory gap (descriptive)

- Token `qwen3pos69mc` runs amendment 69's two arms (e4b `fused_attn4_m_fd`, Unsloth `ckpt_unsloth_m_fv`) once each, at the
  field-device-ratio claim's library commits, with `--mem-census 1` and no profile. The family refuses any window but the claim's 60 steps.
- No predictions. The read reports two quantities side by side and never reads one as evidence about the other:
  - the static census's allocated and reserved bytes at the end of setup, for the device-level gap a timeline read found at the load;
  - the training-window peak and what is live at it, the claim's 26.16 vs 24.27 GB.
- The reducer adds the family, `pos69mc_why` and `pos69mc_block` (self-test 128). `memcensus_block` takes a family, arms and a title, and
  amendment 23's output is unchanged.
