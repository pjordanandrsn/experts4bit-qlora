### Compatibility: transformers 5.19.0 in the Glimmer loader too, with a test both loaders share

- **`arch/glimmer_load.py`** carried the same `compute_default_rope_parameters(cfg, device)` call that #1280 fixed in
  `arch/moe_load.py`. Under transformers 5.19 it raised `TypeError` when rebuilding a meta `inv_freq`. It now passes
  `device=` by keyword and moves the result to the device.
- **`tests/test_rope_rebuild_compat.py`** builds a Qwen3-MoE rotary under `torch.device("meta")` and requires each
  loader's rebuild to return the constructor's own CPU bytes. On transformers 5.19.0, the glimmer case fails without
  this fix and passes with it. Both cases pass on 5.17.0. `test_glimmer_load.py` could not show this, because its
  checkpoint-backed cases skip in CI.
