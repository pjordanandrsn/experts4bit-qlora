### Expert offload is tested under the reentrant checkpoint `enable_fast_train` now defaults to

- `enable_fast_train` routes checkpointed decoder layers through the reentrant checkpoint by default (TC1 amendment 64), and
  only dense-offloaded models are exempt, so expert-offloaded and NVMe-backed models get it too. The offload tests only covered
  `use_reentrant=False`. The three checkpoint tests in `tests/test_offload.py` now run under both checkpoints:
  gradients against a resident reference, one-layer residency through backward, and no eviction when the recompute reaches
  the post-hook. All pass.
- One difference matters. A non-reentrant recompute usually stops before the expert module's post-hook; a reentrant one
  always reaches it. Under the default, every expert-offloaded layer therefore relies on the post-hook's in-backward test
  (`torch._C._current_graph_task_id`). With that test disabled, the non-reentrant gradient and residency tests still
  pass; their reentrant variants fail, as does the post-hook test under both.
