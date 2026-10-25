### bench/common/unsloth_moe_counter.py: one counter for the expert GEMMs Unsloth's MoE QLoRA dispatches

A shared, CPU-tested module that counts Unsloth's expert GEMMs on both of its grouped paths in the release-proven
unsloth_zoo 2026.10.8. Lanes that time Unsloth's MoE training import it, so they cannot count differently.
- **What it counts:**
  - the generic Triton grouped GEMM and its dW (`generic_grouped_mm`, `generic_grouped_wgrad`), wrapped as module attributes so
    every caller is counted. The library's own self-check is counted separately;
  - the NF4-fused grouped GEMM (unsloth-zoo#1935, `maybe_nf4_grouped_mm`), fused or declined;
  - the library's own counters (`GENERIC_CALLS`, `NF4_FUSED_STATS`) as deltas.
- **What binds:** a fused arm binds on `NF4_FUSED_STATS.launches` equal to `expected_launches`, exact: a forward launch per
  fused call plus the dX launches BY STRUCTURE (each expert module backwarded once per micro-batch per step), which holds under
  every checkpointer; the per-call "under grad" split is kept as information only (HF non-reentrant runs two passes under grad
  and backwards one). That one
  equality catches the silent declines (a shape mismatch, a `FAILED_SHAPES` entry) and the failed-launch fallback. An unfused
  arm binds on zero launches and zero calls.
- **The auto arm:** declining above the row cap is its design, so its expected count admits only the calls the library's own
  `nf4_auto_fused` predicate admits, evaluated per call on that call's shapes (`fused_calls_expected`, `check_auto`).
- **The per-call dispatch:** it records Triton-or-torch for every grouped expert GEMM, from the library's own predicate. The
  predicate is wrapped in the package moe_utils at install and, by `patch_loaded(experts)` after the model load, in the module
  object behind the forward each LOADED expert module runs: the zoo binds the forward at patch time, and the compile-cache copy
  can appear during the load. Experts the library hands off by its own predicates (bnb 4-bit, then fp8) are served by the
  package moe_utils, since those modules import it relatively. It records each serving object, each expert type's distinct
  forwards and rule with their module counts (so a mixed model shows), the resolved per-class limits, and attributes every
  decision to the expert-module forward it ran in. `check_dispatch` VOIDs when:
  - a serving object decided no GEMM, a GEMM was decided through an object that serves no forward, or a loaded expert's forward
    holds no predicate;
  - a LoRA-class GEMM decided torch;
  - a Triton-decided call fell back eager;
  - the generic counts differ from the counts the decisions imply (`implied_generic`: the Triton forwards plus the structural
    backward), or one module's training passes inside a step decided differently;
  - optionally, for a lane whose sweep sits inside every limit, a base GEMM decided torch.
- **The environment:** `check_env` refuses any `UNSLOTH_*` key set in the arm's environment that is not registered at its
  value; registered names are held to a scan of every source file of the installed unsloth and unsloth_zoo packages, with the
  count per module recorded (the router switch `UNSLOTH_MOE_FUSED_ROUTER` sits outside the three modules a list would name).
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
    each VOID;
  - the structural count reads right under all four checkpoint shapes (none, HF non-reentrant, reentrant, Unsloth's), and the
    under-grad rule fails under non-reentrant as a mutant; micro-batches multiply the backward; passes that disagree inside a
    step, a step with no training pass, and a pass outside every step each VOID; an eval pass is not a training pass;
  - bnb 4-bit experts on the copy's forward are served by the package (qwen3_5_moe and qwen3_moe stand-ins), and VOID without
    the rule; a 16-bit stack still needs the copy's decisions; a mixed model needs both;
  - the env scan reaches a module no list named; an ambient router switch, an ambient key no file reads and an unscanned
    package each fail `check_env`;
  - absent targets, an absent library and an unreadable auto predicate, which read unread, never pass.
