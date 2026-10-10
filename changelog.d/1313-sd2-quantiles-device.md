### SD2: the box's rotation quantiles on any device (#1313)

- **The bug.** `sd2-5090-2` ended as HARNESS_ERROR (rc 23, $0.713) because `sd2_box._quantiles` passed a CPU
  quantile tensor against angles on the GPU, so V0 crashed after its capture check. The CPU tests could not see it.
- **The fix.** The quantiles are computed on the host. The box's self-test, which runs before the fetch, now also runs
  the rotation summary on the GPU when one is present, so a device mismatch there fails cheaply (rc 21) before any download.
