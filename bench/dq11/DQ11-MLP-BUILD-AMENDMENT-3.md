# DQ11 MLP-adapter BUILD — amendment 3: the version of record, and the runner's terms (#1083)

Prospective, before any Phase 1 data, under `DQ11-MLP-BUILD-PREREG.md` and Amendments 1 and 2, which stay as merged.
The maintainer ruled on 2026-10-19 that this rides in the Phase 1 runner's change and is reviewed with it.

## The version of record: PEFT 0.21.2

Amendments 1 and 2 name "PEFT 0.19.1's `Linear4bit` LoRA forward". 0.19.1 was the seat's development environment, not
the fixture's. Both DQ11 locks pin the same libraries:
- **The locks:** `wheels.json` (science, torch 2.12.1+cu130) and `rehearsal/wheels.json` (A2000, torch 2.11.0+cu128).
- **The pinned libraries:** **PEFT 0.21.2**, transformers 5.5.0, bitsandbytes 0.50.2 and accelerate 1.15.0.

**PEFT 0.21.2 is the version of record:**
- **Basis:** checked against the v0.21.2 tag on 2026-10-19:
  - `Linear4bit.forward` has the same op sequence, and the reviewer found the method body byte-identical to 0.19.1's:
    the base, the clone, the cast, `lora_B(lora_A(x)) * scaling`, the cast back, the add;
  - every attribute the engine checks exists (`cast_input_dtype_enabled`, `use_dora`, `lora_variant`, `set_scale`,
    `scale_layer`);
  - transformers 5.5.0's `MistralMLP.forward` is `down_proj(act_fn(gate_proj(x)) * up_proj(x))`, as before.
- **Refusal:** the runner (`dq11_mlp_phase1_run.py`) refuses any PEFT, transformers, bitsandbytes, accelerate or torch
  that is not the A2000 lock's.
- **What it records:** for each of the five, the installed version, the locked wheel's sha256, the cached wheel's
  sha256, and whether every hashed `.py` file the wheel's RECORD lists is installed with that hash.
- **The two self-checks** run under 0.21.2 on CUDA, so `REF_MISMATCH` is judged against the locked version:
  - the reference with no redraw;
  - the reference through its `_LoraLinear` path with every split off.

## The terms the runner fixes

- **"A: PEFT's own initialization"** (Amendment 1) is the fixture's canonical initializer,
  `adapters/adapter_init.safetensors`. That file is PEFT's `reset_lora_parameters` output (Kaiming A, zero B), and the L
  path installs it after `prepare`. `x` and `gy` are captured with it installed. Since B = 0, that is the frozen base
  model.
- **The L fixture on the A2000** is built as `dq11_arm.prepare` builds L:
  - the same loggetta plan, constraints, workload, seed and TF32 setting;
  - `dense_train.prepare`;
  - the non-reentrant checkpoint check.

  Only `dq11_arm`'s 5090-only device check is left out; the runner's A2000 check stands in its place. `dq11_arm.py` is
  sealed and unchanged.
- **Per-module reads under streaming:** each checked layer is staged by its `dense_offload` handle for its reads and
  evicted after.
- **The step witnesses.** Each is one update of the shipped loop (`train_loop`, the fixture's first batch), each in its
  own process:
  - `step-today`, with the engine never imported;
  - `step-off`, with the engine imported and `E4B_DENSE_MLP_LORA_FUSE` unset;
  - `step-fused`, with it set to 1.
- **The registered "off is bitwise":** `step-off`'s adapters (sha256 of every slot) and loss equal `step-today`'s.
- **Engagement:**
  - `enable` returns 96, with nothing refused;
  - the class-level counter (Amendment 2) sees 0 PEFT LoRA forwards on the MLP projections in `step-fused`, and the same
    q/k/v/o counts as `step-today`;
  - `fallback` is 0, and `calls` is non-zero.
- **"Streaming and checkpointing unchanged":**
  - `step-fused`'s train-prefetch counter deltas equal `step-today`'s;
  - 3 × the fused call count equals `step-today`'s PEFT MLP LoRA forwards. Both include each recompute, so the recompute
    count matches call for call.
- **K8's short run is `K8_UPDATES` = 8 updates**, unfused and fused, each in its own process:
  - from the canonical initializer, on the fixture's first eight batches in order;
  - AdamW at lr 2e-4, constant, clip 1.0;
  - scored by `dq11_common.score` on alpaca-heldout and wikitext-test at the start and after update 8;
  - K8 holds iff |PPL_fused − PPL_unfused| ≤ 0.05 at both points on both texts.
- **The two-AdamW-step state (REPORTED):** two updates of the same loop, from the canonical initializer, unfused, on the
  first two batches.

`PHASE1_PASS` needs the per-module outcome to pass (with any `RANK1_UNRESOLVED` named), together with every witness above
and K8. Anything else is `PHASE1_FAIL`, with each failure named.
