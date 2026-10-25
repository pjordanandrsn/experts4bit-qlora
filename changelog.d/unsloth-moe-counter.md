### bench/common/unsloth_moe_counter.py: one counter for the expert GEMMs Unsloth's MoE QLoRA dispatches

A shared, CPU-tested module that counts Unsloth's expert GEMMs on both of its grouped paths in the release-proven
unsloth_zoo 2026.10.8. Lanes that time Unsloth's MoE training import it, so they cannot count differently.
- **What it counts:**
  - the generic Triton grouped GEMM and its dW (`generic_grouped_mm`, `generic_grouped_wgrad`), wrapped as module attributes so
    every caller is counted. The library's own self-check is counted separately;
  - the NF4-fused grouped GEMM (unsloth-zoo#1935, `maybe_nf4_grouped_mm`), fused or declined;
  - the library's own counters (`GENERIC_CALLS`, `NF4_FUSED_STATS`) as deltas.
- **What binds:** a fused arm binds on `NF4_FUSED_STATS.launches` equal to the expected fused base GEMMs, exact. That one
  equality catches the silent declines (a shape mismatch, a `FAILED_SHAPES` entry) and the failed-launch fallback. An unfused
  arm binds on zero launches and zero calls.
- **The auto arm:** declining above the row cap is its design, so its expected count admits only the calls the library's own
  `nf4_auto_fused` predicate admits, evaluated per call on that call's shapes (`fused_calls_expected`, `check_auto`).
- **The environment:** `check_env` refuses any unregistered `UNSLOTH_MOE_*` key or any value other than the arm's registered pin.
  Wrapper counts bind only beside a counterpart that counts per call even when Python does not run.
- **Tests** (`tests/test_unsloth_moe_counter.py`), on CPU stand-ins at the real module paths:
  - a VOID for each silent decline and for a failed launch;
  - the fused path's bypass of the generic counter;
  - a local alias bound before install, missed and caught by the library cross-check;
  - an exported (captured) block that undercounts the wrapper while the in-op counter counts, caught by `check_equal`;
  - auto: a forced decline at an eligible shape VOIDs; a launch at an ineligible shape VOIDs; an ambient
    `UNSLOTH_MOE_NF4_AUTO_MAX_ROWS` VOIDs by `check_env`; the count follows the library's own constant;
  - a failed self-check leaves the generic counts at zero, so a lane's bar fails closed;
  - absent targets, an absent library and an unreadable auto predicate, which read unread, never pass.
