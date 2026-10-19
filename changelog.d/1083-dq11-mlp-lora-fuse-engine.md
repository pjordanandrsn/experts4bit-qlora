### DQ11's dense MLP LoRA fusion engine, off by default (#1083)

`experts4bit_qlora/engines/dense_lora_fuse.py` is the engine that `bench/dq11/DQ11-MLP-BUILD-PREREG.md` registered
(D1, D2). `enable_dense_mlp_lora_fuse(model)` takes each decoder MLP whose `gate_proj`/`up_proj`/`down_proj` are
vanilla PEFT LoRA modules on bitsandbytes NF4 bases:
- **gate and up:** one FP32 cast of their shared input and one rank GEMM against the stacked `[A_gate; A_up]`;
- **every adapter:** its B GEMM added into its base output in FP32 by one in-place `addmm_` (the scale as alpha), cast
  back once;
- **autograd:** a custom Function returns the gradients of the input, every A and every B, and checkpoint recompute
  runs the same Function.

Rank compute stays FP32, so outputs and gradients equal PEFT's to rounding. The registration's Phase 1 holds each tensor
to the unfused path's neutral floor on the A2000.

It returns the adapters fused (96 on Mistral-7B). An MLP with dropout, DoRA, more than one active adapter, a merged or
non-FP32 adapter, a base that is not NF4, or a biased base keeps PEFT's path, and its reason is in
`DENSE_MLP_LORA_STATS["refused"]`. `disable_dense_mlp_lora_fuse` restores every fused MLP's own forward.

No trainer calls it. `E4B_DENSE_MLP_LORA_FUSE` (read by `dense_mlp_lora_fuse_requested()`) is off unless set; DQ11's
Phase 1 harness and Phase 2 kit read it. No default changes.
