### Changed

- The private-marker guard adds a second added-lines-only rule for a GPU's UUID, the form torch prints
  (`torch.cuda.get_device_properties()`: `uuid=` and the 8-4-4-4-12 hex, without nvidia-smi's `GPU-` prefix), with its
  canary line. It matches only after `uuid=`, so run ids, commit shas and nonces never trip it.
