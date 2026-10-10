### Training routing trace: the capture wrapper no longer fails at startup

`bench/train-routing-trace/run_capture.py` reached the optimizer step hook as `torch.optim.optimizer.<name>`. That
submodule is not an attribute of `torch.optim` after a plain `import torch`, so the wrapper raised `AttributeError`
before loading a model.
- It now imports `register_optimizer_step_post_hook` from `torch.optim.optimizer`.
- `environment()` records transformers as unknown if it is absent, instead of failing the save.
- A new CPU test drives `run_capture.main()` end to end against a stub harness. It fails on the previous code with the
  same error.

No capture had run, so no data is affected.
