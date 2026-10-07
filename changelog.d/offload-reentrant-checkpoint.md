### Expert offload is tested under the reentrant checkpoint

- `enable_fast_train`'s reentrant-checkpoint default also applies to expert-offloaded and NVMe models. The offload
  checkpoint tests now run under both checkpoint kinds, and pass. A reentrant recompute always reaches the expert
  module's post-hook, so the tests now also catch a broken in-backward check there.
