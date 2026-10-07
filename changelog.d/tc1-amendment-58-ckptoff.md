### TC1 amendment 58 registered: checkpoint inputs in pinned host memory on packed rows (P153-P157)

- Token `qwen3ckptoff` runs e4b's matched arm on packed rows at its defaults, `E4B_CKPT_OFFLOAD` 0 against 1, two draws each, with
  Unsloth's matched arm beside it and peaks split by phase.
- P153: the training-phase peak falls by 0.6 GB or more. P154: o1 / o0 at most 1.05. P155: held-out within 0.005. P156 / P157: o0 / o1
  within 1.7 / 1.0 GB of Unsloth's training-phase peak.
- `tc1_arm.py` records `ckpt_offload_layers` and `ckpt_offload_env`. The reducer adds the family, `ckptoff_why` and `score_ckptoff`.
  Self-test 123.
