### Fix: every residency state class serves the MoE residual, or never sees it (#1313, found by `p127-prove-1`)

- **The bug.** #1477's patched experts forward handed `residual=` (even `None`) to the residency state's `forward`.
  hybrid's `_HybridTier`, the state class of the served all-resident build, overrides `forward` without it. So since
  #1477, every MoE call of `serve_paged`'s default GPU build raised a TypeError, the licence probe in `build_engine`
  first. The CPU tests had used a toy experts module. P127's proving rental caught it before any reading.
- **The fix.**
  - `hot_residency._state_forward` passes no `residual=` keyword when there is none. It passes the keyword where the
    state class takes it (signature, read once per class), and otherwise the residual is the layer's own add.
  - `_HybridTier.forward` takes `residual=` and hands it to the base forward, so the fold engages on the served path.
  - The base forward's non-collapsed residual path calls `_HotResidency.forward` by name. A subclass override (hybrid's
    prefetch submit and amortization count) therefore runs once per call, not twice.
  - `glue_r2.license_moe_residual` turns a probe that raises into a refused licence, with the error in
    `info["moe_residual"]["probe_errors"]`. A probe can no longer stop a build.
- **Tests.**
  - The real `_HybridTier` signature and its pass-through.
  - A state class without `residual=`.
  - The no-re-entry path.
  - A guard: every `_HotResidency` subclass in the package that overrides `forward` takes `residual=`.
  - A probe that raises.
  - Each test is mutation-checked.
