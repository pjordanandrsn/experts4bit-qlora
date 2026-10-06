### Serve estimate: the bulk KV flush is priced (lane SV5's out-of-memory)

- Lane SV5 (#1242, RTX 4090) served Qwen3-30B-A3B all-VRAM at 8 × 8192, the plan a planner made from SV4's receipts.
  It ran out of memory at 8,000-token prompts with 22.59 GiB allocated, against an estimate of 22.150 GiB.
- The estimate listed the bulk KV flush (`E4B_PAGED_BULK_KV`, on by default) as not modelled. At a finished prompt it
  quantizes every pool layer's K/V and holds it until one bulk write. At a full 8,192-token slot that is 526 MiB on
  Qwen3-30B-A3B.
- `fp8_paged_kv.append_prompt_peak_bytes(layer_geometry, T)` is now a pure function, and
  `Fp8PagedKV.append_prompt_peak_bytes` calls it.
- The estimate prices it as a derived item at the slot's capacity, the bound the server's prefill-graph headroom check
  already used. `ServeSetup.bulk_kv` (default `True`) sets `E4B_PAGED_BULK_KV`.
- SV5's estimate becomes 22.664 GiB, above the 22.62 GiB the out-of-memory arm had reached.
