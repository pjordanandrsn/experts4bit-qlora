### SD2 reducer (#1313): the depth report reads `v_angles.pt` with `weights_only=True`

- **Why.** Re-derivations keep using `bench/sd2/sd2_reduce.py` after the lane closed, and they load a receipt's
  `v_angles.pt`. Asking for `weights_only=True` explicitly keeps that load from building pickled objects even where
  torch's own default is off (before torch 2.6, or under `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`).
- **Unchanged.** `sd2-5090-3`'s depth report reproduces exactly from its `v_angles.pt` (24 records). The verdict
  never read this file. `staged.sha256` re-pins the reducer.
