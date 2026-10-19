### The dense MLP LoRA fusion follows PEFT state changes made after fusing (#1083)

`engines/dense_lora_fuse.py` (#1809) now re-reads each fused MLP's PEFT state on every call. If a projection's adapters
were disabled (PEFT's `disable_adapter()` context), another adapter was set, an adapter was merged, or PEFT's input
cast was turned off after fusing, that call runs the class's own forward. So does a call under autocast. Each such call
is counted in `DENSE_MLP_LORA_STATS["fallback"]`, which replaces the `autocast` counter.

- **New refusals:** an MLP whose input cast is already off at `enable` time, and a hook on a projection's `lora_A` or
  `lora_B`, or a backward hook on the projection, which the fused forward would skip.
- **Behavior:**
  - a second `enable` counts the MLPs it already fused;
  - the backward skips the input-gradient matmul when the input needs no gradient;
  - the `calls` note says it counts checkpoint recomputes.
- **Tests:** a float64 `gradcheck` through the checkpoint recompute.

It is still off by default.
