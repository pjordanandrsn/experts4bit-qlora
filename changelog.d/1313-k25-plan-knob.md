### Added

- **`E4B_K25_PLAN`** (#1313): K25's block shapes on the NF4 store's device-grouped decode route, per projection.
  `today` is the route's plan for both projections, the default (also when unset), unchanged. `c1` is P156's first
  carried pair (gate_up `{block_n 64, kc 256, warps 4, stages 2}`, down `{block_n 128, kc 128, warps 8, stages 3}`), which
  lane P160 read at 1.6598x today's plan under K39's decode in a microbench at Qwen3-30B-A3B's W16 shapes
  (`bench/p160`).
  - **Opt-in:** no default changes, and no served speed claim. A served read of its own decides any default.
  - **Only under `prmt`:** the select tree at `c1`'s shapes is slower than at today's (P155), so `c1` under any decode
    other than `E4B_K25_LUT`'s `prmt` (explicit, or qwen3_moe's default where it resolves) is refused, never run. It is
    also refused with K25 off or absent, also under `E4B_NF4_GROUPED_SMALLM=auto`. Any other value is refused.
  - **At startup:** `serve_paged.build_engine` refuses what no family can change before reading the model's config, and
    the rest once the family's decode is resolved, before any weight loads (`hot_residency.resolve_k25_plan`);
    `info["k25_plan"]` records the resolution and each projection's shape.
  - **Engagement record:** `hot_residency.K25_PLAN_SEEN` counts K25's calls by plan (`"<plan>|le256"`; eager calls and
    graph captures, not replays), and `/health`'s `prefill_routes()["seen"]["moe_k25_plan"]` carries it.
  - **Bitwise on a GPU:** through the served route's own call, the tree, `prmt`, and `prmt` with `c1` give the same bits
    at B=16 (both glues), T == 1 and a CUDA-graph replay, including a K that neither of `c1`'s KCs divides
    (`tests/test_k25_lut_gpu.py`).
