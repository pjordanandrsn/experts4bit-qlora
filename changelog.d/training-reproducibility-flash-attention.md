### Docs: training gradients are not bitwise reproducible under flash attention

- The limitations shared by every solution page now say that training
  gradients on GPU differ from run to run because PyTorch's default
  flash-attention backward is nondeterministic, that neither
  `CUBLAS_WORKSPACE_CONFIG` nor `torch.use_deterministic_algorithms(True,
  warn_only=True)` removes it, and that the math attention backend
  (`sdpa_kernel(SDPBackend.MATH)`) gives bitwise-reproducible gradients at a
  cost in speed and memory. Documentation only; no code change.
