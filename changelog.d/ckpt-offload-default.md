### `E4B_CKPT_OFFLOAD` is on by default under `enable_fast_train` (TC1 amendment 59); `0` turns it off

- `enable_fast_train` now keeps every checkpointed decoder layer's input in pinned host memory by default (the reentrant checkpoint
  inside `save_on_cpu`, #1298). Unset or `1` is on, and `0` keeps Hugging Face's checkpointing with the inputs on the GPU.
- The evidence, Qwen3-30B-A3B on one RTX 5090 in torch 2.12:
  - packed 4,096-token rows (TC1 amendment 58, `e4b.train.ckpt-offload.packed-4k.5090.2026-10-07`): the training-phase peak 0.739 GB
    lower, at 1.003 of the step;
  - TC1's field recipe, matched and shipped arms (amendment 59, `e4b.train.ckpt-offload.field.5090.2026-10-07`): 0.948 and 0.916 of
    the step, the matched training-phase peak 0.171 GB lower.
  Held-out stayed within 0.003 everywhere. Gradients equal the default checkpointing's exactly in the tests, so a held-out difference
  across the change is run-to-run noise, not the switch.
- Why the field-recipe step got faster is not established yet: TC1 amendment 62 reads the reentrant checkpoint alone
  (`E4B_CKPT_OFFLOAD=reentrant`, new, a diagnostic) against the copies.
- The default leaves a model alone whose decoder layers carry `enable_dense_offload`'s handles, because that pairing is untested. Only an
  explicit `E4B_CKPT_OFFLOAD=1` pairs them, and `enable_dense_offload` warns when it finds offloaded checkpoints. Without gradient
  checkpointing the default is silent. The CLI trainer keeps its own checkpointing and is unchanged.
- Scope: one model, one card class, torch 2.12, two recipes.
