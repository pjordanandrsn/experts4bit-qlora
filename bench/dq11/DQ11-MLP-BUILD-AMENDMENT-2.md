# DQ11 MLP-adapter BUILD — amendment 2: how the zero-PEFT-forward witness is counted (#1083)

Prospective, before any Phase 1 data, under `DQ11-MLP-BUILD-PREREG.md` and Amendment 1, which stay as merged. The
maintainer ruled on 2026-10-19 that this erratum be a new amendment and ride in the Phase 1 harness's change.

## The erratum

Amendment 1 says the witness "zero PEFT LoRA MLP forwards in a step" is "counted by hooks on the LoRA modules". It is
counted instead by a **class-level wrapper** on PEFT's LoRA `forward`, the `Linear4bit` LoRA class on the fixture
(`lora_forward_counter` in `dq11_mlp_phase1.py`). That wrapper is restored on exit, and it never adds a module hook.

**Why.** Since the engine follow-up (#1814), a hook on a projection or on its `lora_A` / `lora_B` added after fusing
makes that MLP's call fall back to the class's own forward, which is PEFT's path. A hook-counted witness would therefore
measure the fallback, not the fused path.

## What the harness asserts on the real fixture

- `enable_dense_mlp_lora_fuse` returns **96**, with nothing refused (`ENGAGE_COUNT` otherwise).
- `DENSE_MLP_LORA_STATS["fallback"]` is **0** after every fused read and every fused step. A fallback that fires is a
  VOID (`VOID_FALLBACK`), never a quiet pass of the class's path.
- The wrapper counts **0** PEFT LoRA forwards on the 96 MLP projections in a fused step. On the q/k/v/o projections it
  counts the same as the unfused step.
