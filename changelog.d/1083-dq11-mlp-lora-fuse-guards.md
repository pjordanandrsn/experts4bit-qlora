### The dense MLP LoRA fusion follows PEFT state changes made after fusing (#1083)

`engines/dense_lora_fuse.py` (#1809) now re-reads each fused MLP's PEFT state on every call. That call runs the class's
own forward if, after fusing, any of these happened to a projection:
- its adapters were disabled (PEFT's `disable_adapter()` context);
- another adapter was set, or an adapter was merged;
- PEFT's input cast was turned off;
- a hook was added to it or to its `lora_A` / `lora_B` (PEFT's mixed-batch `adapter_names` arrive that way);
- its scale changed so the three differ (`set_scale`, `scale_layer`);
- an adapter module was replaced under the same name.

A call under autocast also runs the class's own forward. Each such call is counted in
`DENSE_MLP_LORA_STATS["fallback"]`, which replaces the `autocast` counter. A changed scale that all three projections
share is read live, as PEFT reads it.

- **New refusals:** an MLP whose input cast is already off at `enable` time, and a hook on a projection's `lora_A` or
  `lora_B`, or a backward hook on the projection, which the fused forward would skip.
- **Behavior:**
  - a second `enable` counts the MLPs it already fused;
  - the backward skips the input-gradient matmul when the input needs no gradient;
  - the `calls` note says it counts checkpoint recomputes.
- **Tests:** a float64 `gradcheck` through the checkpoint recompute.

It is still off by default.
