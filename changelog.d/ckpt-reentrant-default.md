### `enable_fast_train` checkpoints with PyTorch's reentrant checkpoint by default (TC1 amendment 64); `E4B_CKPT_OFFLOAD=0` is the way back

- Every checkpointed decoder layer now runs PyTorch's reentrant checkpoint instead of Hugging Face's non-reentrant one, unless
  `E4B_CKPT_OFFLOAD=0`. `E4B_CKPT_OFFLOAD=1` adds the host-memory inputs (the offload, still opt-in), and `=reentrant` names the default
  explicitly.
- Why, on Qwen3-30B-A3B on one RTX 5090 (`e4b.train.ckpt-flavour.default.5090.2026-10-07`, `e4b.train.ckpt-flavour.field.5090.2026-10-07`):
  the step falls to 0.980 of Hugging Face's checkpoint on packed 4,096-token rows (torch 2.12), 0.900 at TC1's field recipe (torch 2.12)
  and 0.838 there in torch 2.8 on a host-bound box. The training peak is unchanged, and held-out stayed within 0.004. Scope: one model,
  one card class, torch 2.12 and 2.8, two recipes.
- **What changes for a caller.** A reentrant checkpoint does not support `torch.autograd.grad` or `backward(inputs=...)` through the
  checkpointed layers. It gives a layer gradients for its contents only when the layer's input requires grad, so `enable_fast_train`
  now calls the model's `enable_input_require_grads()` by default (as PEFT does for reentrant checkpointing; no value changes). Set
  `E4B_CKPT_OFFLOAD=0` to keep Hugging Face's checkpoint.
- Left alone by default: a model without gradient checkpointing (silently), and a model whose decoder layers carry
  `enable_dense_offload`'s handles (that pairing is untested; an explicit `1` or `reentrant` pairs them, and `enable_dense_offload` warns).
  The CLI trainer keeps its own checkpointing.
- The offload's trade, for long rows: 0.74 GB of training peak for 1.023 of the reentrant step on packed rows (P180 FALSIFIED against
  1.01), so it is not recommended by default.
