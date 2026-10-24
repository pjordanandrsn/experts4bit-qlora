### Changed

- The private-marker guard adds a second added-lines-only rule for a GPU's UUID without nvidia-smi's `GPU-` prefix: as
  torch prints it (`torch.cuda.get_device_properties()`: `uuid=` and the 8-4-4-4-12 hex) and as a receipt records it
  (`"uuid": "..."`, `uuid='...'`, `uuid = ...`), with canary lines for both shapes. It matches only after a uuid key
  and its separator, so run ids, commit shas, nonces and other bare hex never trip it.
