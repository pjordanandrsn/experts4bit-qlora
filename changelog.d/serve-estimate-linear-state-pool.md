### Serve estimate: the hybrid models' linear-attention state pool is priced

- The paged server keeps a per-slot conv window and recurrent state for every Gated DeltaNet layer (Qwen3.5 / 3.6 /
  Next, `engines.linear_state.LinearStatePool`). The estimate listed it as "recurrent state ... not modelled".
- It is now a derived device item, `linear_state_pool_bytes(topology.linear_state_layers, slots)`. Per layer and slot:
  - a `[conv_dim, conv_kernel]` conv window in the model's dtype;
  - a `[v_heads, head_k_dim, head_v_dim]` recurrent state in fp32, which is how transformers keeps it.
  - Slots are `max_seqs` plus the decode graphs' scratch slots, as the runner sizes the pool.
- `MoETopology.linear_state_layers` comes from `engines.linear_state.state_geometry`, read from the very modules the
  pool drives.
- A test stores a real Gated DeltaNet forward's state into a real pool and checks `nbytes()` against the price,
  74,240 B both on a tiny model.
- Qwen3.6-35B-A3B at 16 sequences with decode graphs: **1.93 GiB** (30 layers × 32 slots), previously unpriced.
