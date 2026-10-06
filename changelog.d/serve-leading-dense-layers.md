### `serve_paged` serves models whose first layers are dense (ERNIE-4.5; DeepSeek-V2's dense-first layout too, but its MLA attention is refused, #1233)

ERNIE-4.5-21B-A3B (layer 0 dense, MoE in layers 1–27) failed twice in `build_engine`.
- **The arena's layer ids.**
  - A bake keys its rows by the checkpoint's layer numbers, and the server stamped its MoE modules with ordinals
    0..L-1, so the first module asked for row (0, 0).
  - `serve_paged.arena_layer_ids(arena, L)` reads the arena's own ids and refuses one whose layer count is not the
    model's.
  - The server re-keys its placement manifest to those ids and passes them as `enable_hybrid_tier(layers=...)`.
- **The KV pool's layer count.**
  - The pool was sized by the MoE layers (27), but layer 0 has attention too, and the 28th append indexed past the
    pool.
  - `engines.paged_runner.decoder_layers(config)` now gives the decoder's layer count for both the server's pool and
    `MoETopology.kv_layers`, so the serve estimate prices the same pool.
- Measured on an RTX A2000: ERNIE-4.5-21B-A3B now builds and serves 4 × 1,024-token prompts with the solver's
  tiers.
