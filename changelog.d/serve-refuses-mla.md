### `serve_paged` and its estimate refuse multi-head latent attention before loading

- DeepSeek-V2-Lite was planned as a feasible serve. The FP8 paged pool keeps one head dim per layer for K and V, and
  it was built `head_dim` (64, the rotary width) wide.
- MLA hands attention keys of `qk_nope + qk_rope` (192) and values of `v_head_dim` (128). The first prompt's append
  refused them, after the whole model had loaded.
- `engines.paged_runner.kv_layout_refusal(config)` names it from the config (`kv_lora_rank`).
- `build_engine` now refuses before reading a weight, and `paged_state_refusal` (and so the serve estimate) refuses
  with the same words.
