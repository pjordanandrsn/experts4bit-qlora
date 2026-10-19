### DQ11 MLP-adapter BUILD Phase 1: the per-module reads, reference and bar (#1083)

`bench/dq11/dq11_mlp_phase1.py` holds the per-module half of Phase 1 (`DQ11-MLP-BUILD-PREREG.md` D3, Amendment 1),
tested on CPU against real bitsandbytes NF4 bases:
- **The truth:** `peft_read`, PEFT's own unfused path, gives eleven tensors: three outputs, dx, dh, and six A/B
  gradients.
- **The reference:** `ref_read` repeats PEFT 0.19.1's `Linear4bit` LoRA forward op for op, with redraws R1–R5. With no
  redraw it is bit-identical to PEFT; that is the `REF_MISMATCH` check.
- **The engine:** `fused_read` reads the engine on the same module.
- **The rest of the read:** the 31-draw floor and its coverage map, Amendment 1's adapter terms and shares, seeded B, the
  σ choice, `e_min`, the bar, and the outcome line (`PHASE1_FAIL`, or `PHASE1_PASS` with any `RANK1_UNRESOLVED`
  tensors named);
- `lora_forward_counter`, the zero-PEFT-forward witness, counts with a class-level wrapper rather than module hooks.

`DQ11-MLP-BUILD-AMENDMENT-2.md` registers that erratum to Amendment 1 before any Phase 1 data. Since #1814, a hook
makes a fused MLP fall back to PEFT's path. The amendment also registers the `enable == 96` and `fallback == 0` asserts
(`ENGAGE_COUNT`, `VOID_FALLBACK`).

On CPU, R4 (the rank-split LoRA input gradient) and R1's forward change are absorbed by the bf16 cast. So dh is moved by
no redraw, and the bar requires it bit for bit, as the registration provides. The A2000 runner is a separate change.
