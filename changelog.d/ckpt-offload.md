### `E4B_CKPT_OFFLOAD=1`: checkpointed decoder layers keep their input in pinned host memory (opt-in)

- Under gradient checkpointing every decoder layer keeps its input hidden states on the GPU until its backward. On Qwen3-30B-A3B's packed
  4,096-token rows that is 47 × 16.8 MB, 0.79 GB of the training peak. TC1 amendment 57's census found it the largest single group above
  Unsloth's (`e4b.train.memory.packed-4k-train-census.5090.2026-10-07`).
- With the variable set, `enable_fast_train` routes each checkpointed layer through PyTorch's reentrant checkpoint inside
  `torch.autograd.graph.save_on_cpu(pin_memory=True)`. That saves exactly the layer's input, in host memory. It also calls
  `enable_input_require_grads()`, which reentrant checkpointing needs, and which changes no value. `disable_fast_train` undoes it, and
  layers that `E4B_MOE_KEEP_LAYERS` keeps are left alone.
- Gradients equal Hugging Face's default checkpointing exactly on CPU, including with a frozen embedding. e4b's fused ExpertsLoRA path
  under the offloaded checkpoint matches the plain path exactly on CUDA (RTX A2000). The copies are synchronous: what they cost a
  training step is for a TC1 box to read, so it stays opt-in.
- **Untested with `enable_dense_offload`** (maintainer review): the dense-weight offload's train-prefetch schedule is documented for the
  non-reentrant checkpoint's layer order, and this switches checkpointed layers to the reentrant one. The layer order should match
  (forward 0..L-1, then each recompute L-1..0), but no test runs the two together. Read that pairing before relying on it.
