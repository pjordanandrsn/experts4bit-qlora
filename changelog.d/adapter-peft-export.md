### adapter_peft: export a trained adapter to the PEFT layout SGLang's MoE LoRA loader reads

`python -m experts4bit_qlora.adapter_peft ADAPTER_DIR OUT_DIR` writes a verified adapter directory as PEFT LoRA files
that SGLang v0.5.23's MoE LoRA loader reads, plus a provenance record. Nothing in e4b's own serving changes.
- **Copied, not recomputed:** the expert stacks keep their `[E, ., .]` shapes under
  `<stack>.{gate_up_proj,down_proj}.lora_{A,B}.weight`, which SGLang reads as per-expert factors. Attention keeps its
  projections. `lora_alpha` and `r` are the setup's.
- **Refused, naming the reason:**
  - an unverified directory, or no base revision;
  - unknown tensors, or an incomplete stack;
  - a non-gated stack, or a one-expert stack (SGLang would read that as its shared-outer form);
  - a recorded scale other than alpha/r;
  - an output path that is not empty.
- **Tested on the CPU against the trained forward:** the parity test recomputes `ExpertsLoRA`'s output from the exported
  files in SGLang's documented layout. Gate/up swapped, experts permuted, alpha doubled and the down delta dropped each
  fail it.
- **Not read:** serving parity on SGLang itself. The export is exact only on a served base equal to training's
  dequantized NF4 experts (docs/ADAPTER_PORTABILITY.md §7).
