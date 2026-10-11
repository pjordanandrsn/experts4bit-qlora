### Added

- `bench/q235/instrument/q235_compare.py` reports, beside the per-layer error of the residual state, each layer's error on its own
  contribution (`update_rel_err`: ||d_a - d_b|| / ||d_b|| with d_i = h_i - h_(i-1)). Deep in a model a layer adds a few percent
  to the residual, so an error confined to one layer barely moves the state there; its contribution error reads it at full
  size. A known-answer test pins both readings (#835).
