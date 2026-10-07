### TC1 amendment 59 registered: checkpoint inputs in pinned host memory at the field recipe (P158-P161)

- Token `qwen3ckptofff` runs e4b's shipped and matched arms at TC1's field recipe, `E4B_CKPT_OFFLOAD` 0 against 1, two draws a side in
  A B B A order, with peaks split by phase. This is the read amendment 58's rule names before any default.
- P158 / P159: f1 / f0 at most 1.01 on the matched / shipped arm. P160: held-out within 0.005 on each arm. P161: the matched arm's
  training-phase peak falls by 0.10 GB or more.
- `ckptoff_why` takes the field recipe's form (padding and loss stay stock there). The reducer adds the family and `score_ckptofff`.
  Self-test 124.
