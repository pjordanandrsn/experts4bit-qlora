### bench/common/unsloth_moe_counter.py: one counter for the expert GEMMs Unsloth's MoE QLoRA dispatches

A shared, CPU-tested module that counts Unsloth's expert GEMMs on both of its grouped paths in the release-proven
unsloth_zoo 2026.10.8. Lanes that time Unsloth's MoE training import it, so they cannot count differently.
- **What it counts:**
  - the generic Triton grouped GEMM and its dW (`generic_grouped_mm`, `generic_grouped_wgrad`), wrapped as module attributes so
    every caller is counted. The library's own self-check is counted separately;
  - the NF4-fused grouped GEMM (unsloth-zoo#1935, `maybe_nf4_grouped_mm`), fused or declined;
  - the library's own counters (`GENERIC_CALLS`, `NF4_FUSED_STATS`) as deltas.
- **What binds:** a fused arm binds on `NF4_FUSED_STATS.launches` equal to the expected fused base GEMMs, exact. That one
  equality catches the three silent declines and the failed-launch fallback. An unfused arm binds on zero launches and zero calls.
  Wrapper counts bind only beside a counterpart that counts per call even when Python does not run.
- **Tests** (`tests/test_unsloth_moe_counter.py`), on CPU stand-ins at the real module paths:
  - a VOID for each silent decline and for a failed launch;
  - the fused path's bypass of the generic counter;
  - a local alias bound before install, missed and caught by the library cross-check;
  - an exported (captured) block that undercounts the wrapper while the in-op counter counts, caught by `check_equal`;
  - absent targets and an absent library, which read unread, never pass.
