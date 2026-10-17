### serve_paged refuses an unservable hybrid before reading any weight

serve_paged gave the same verdict on hybrid models before; it now gives it earlier. Before any weight is read, it asks
the paged runner's own verdict, `paged_runner.paged_state_refusal`, on the empty-weight module tree that transformers
builds from the config (`config_state_refusal`). The serve estimate and the runner's `layer_plan` ask the same
function, so the three cannot disagree.

**Which models it refuses.** Hybrids whose non-attention layers the runner keeps no state for:
- Nemotron-H and Jamba (Mamba layers, which transformers labels `linear_attention`, while the per-slot pool drives
  Gated DeltaNet only);
- granite-4.0-h (Mamba);
- LFM2 (short convolutions).

These now fail in seconds with the reason. Before, a full weight load (about 120B parameters for Nemotron-3-Super)
came first. Multi-head latent attention is refused inside the same verdict, as before.

**A config without `layer_types` is no longer assumed to be all attention** when it declares other mixers:
- a per-layer list under `layers_block_type`, `hybrid_override_pattern` or `attn_layer_period` /
  `attn_layer_offset`;
- or `mamba_*` / `ssm_*` geometry.

transformers 5.17 derives `layer_types` for Nemotron-H and Jamba, but e4b's floor is transformers>=5.0, and a config
that reaches the runner without them used to plan every layer as attention.

`tests/test_serve_state_refusal.py` tests the committed fixture configs: digests are checked, the hybrids are refused
before load, seven served families are admitted, and both Nemotron-H spellings are refused by the guard. Training paths
are unchanged.
