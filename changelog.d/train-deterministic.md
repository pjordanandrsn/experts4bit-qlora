### Added

- `E4B_TRAIN_DETERMINISTIC=1` (opt-in, `engines/train_determinism.py`): the fused training path's deterministic mode. It
  turns on the deterministic token gather (`E4B_TRAIN_DET_GATHER=1`), `torch.use_deterministic_algorithms(True)` scoped
  to each backward only (try/finally; the forward and its loss run with it off), and
  `torch.utils.deterministic.fill_uninitialized_memory = False`. It refuses by name at trainer start unless
  `CUBLAS_WORKSPACE_CONFIG` is `:4096:8` or `:16:8`. The mode is this knob PLUS a self-identity check in the gate's own
  run (two blocks of three steps, every trainable gradient bitwise), never the flag alone: the fill-off piece is safe
  only while no kernel reads memory it did not write. No default changes; no cost is stated here.
