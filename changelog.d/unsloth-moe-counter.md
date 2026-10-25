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
- **The per-call dispatch:** it records Triton-or-torch for every grouped expert GEMM, from the library's own predicate. The
  predicate is wrapped in the package moe_utils at install and, by `patch_loaded(experts)` after the model load, in the module
  object behind the forward each LOADED expert module runs: the zoo binds the forward at patch time, and the compile-cache copy
  can appear during the load. It records each serving object, each expert type's distinct forwards with their module counts (so
  a mixed model shows) and the resolved per-class limits. `check_dispatch` VOIDs when:
  - a serving object decided no GEMM, a GEMM was decided through an object that serves no forward, or a loaded expert's forward
    holds no predicate;
  - a LoRA-class GEMM decided torch;
  - a Triton-decided call fell back eager;
  - the generic counts differ from the counts the decisions imply (`implied_generic`);
  - optionally, for a lane whose sweep sits inside every limit, a base GEMM decided torch.
- **The environment:** `check_env` scans the three library modules' source for every `UNSLOTH_*` key they read. Each one is
  unset or at the arm's registered pin; a prefix test would miss `UNSLOTH_DISABLE_MOE_TRITON` and others.
  Wrapper counts bind only beside a counterpart that counts per call even when Python does not run.
- **Tests** (`tests/test_unsloth_moe_counter.py`), on CPU stand-ins at the real module paths:
  - a VOID for each silent decline and for a failed launch;
  - the fused path's bypass of the generic counter;
  - a local alias bound before install, missed and caught by the library cross-check;
  - an exported (captured) block that undercounts the wrapper while the in-op counter counts, caught by `check_equal`;
  - auto: a forced decline at an eligible shape VOIDs; a launch at an ineligible shape VOIDs; an ambient
    `UNSLOTH_MOE_NF4_AUTO_MAX_ROWS` VOIDs by `check_env`; the count follows the library's own constant;
  - a failed self-check leaves the generic counts at zero, so a lane's bar fails closed;
  - dispatch: a cache-copy forward is counted, and one appearing after install is missed and VOIDs unless `patch_loaded` reads
    it off the loaded experts; experts on a forward other than the fresh `get_forward_moe_backend()` answer are counted when
    passed and VOID when not; a mixed model records each forward and VOIDs until every one decides; experts left on the original
    forward VOID; a decorated forward unwraps; the wrapper returns the predicate's own answer and uninstall restores it; a LoRA
    call deciding torch, an eager fallback, a backward off the Triton branch and (with the guard) a base GEMM deciding torch
    each VOID; a checkpointed double forward is counted twice; an ambient `UNSLOTH_DISABLE_MOE_TRITON=1` and an unscanned library each fail
    `check_env`;
  - absent targets, an absent library and an unreadable auto predicate, which read unread, never pass.
