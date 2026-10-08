### Decode folds: the rotary and the router-weight cast are licensed on what the module computes (fix; lanes P115, FAM)

- **What changed.**
  - **The rotary.** `glue_r2`'s three attention folds and `fuse_qkv` now probe the module's own rotary, the
    `apply_rotary_pos_emb` its class's forward calls. The fold applies only where that function is rotate-half, which
    is what the kernels compute, for arbitrary cos/sin (`glue_r2.rotary_is_rotate_half`). A module whose rotary
    differs, cannot be read, or raises on the probe keeps its own forward. `fuse_qkv` raises on a `Qwen3MoeAttention`
    whose rotary is not rotate-half: transformers drifted.
  - **The router-weight cast.** The router epilogue now casts its routing weights to the logits' dtype only where the
    router's own forward returns them in that dtype, as the probe reads it from the module. That holds under every
    `E4B_ROUTER_EPI_CAST` setting. `/health`'s fold report counts the routers that keep fp32 (`fp32_upstream`).
- **Why.** Both were found by lane FAM's inventory (#1362, #1368):
  - ERNIE-4.5's attention has exactly the q/k/v/o structure the rope-only fold accepts, but interleaves its rotary.
    The fold applied rotate-half, and the rotated q came out about 1.1 off in relative norm.
  - Mixtral's router returns fp32 weights. Casting them to bf16 changed upstream's function instead of matching it.

  Neither fold is on by default for either family. Explicit `auto` or `1` reached both.
- **Who is affected.** Only explicit `auto` / `1` on those families:
  - ERNIE-4.5's attention is no longer folded;
  - Mixtral's fused router keeps fp32 weights.

  Qwen3-MoE, Granite-MoE and gpt-oss fold and cast exactly as before.
- **Tests:**
  - `tests/test_glue_r2.py`:
    - the probe on rotate-half and interleaved fixtures;
    - each attention fold refusing an interleaved rotary;
    - transformers' own ERNIE-4.5 MoE attention refused (forced past the guard, it changes its output and its rotated
      q);
    - transformers' Mixtral, GraniteMoe and Qwen3-MoE attention still licensed and matching.
  - `tests/test_router_epilogue.py`: a router returning fp32 is never cast; transformers' Mixtral router keeps fp32,
    and Qwen3-MoE's casts to bf16, under the default and under `1`.
