### Serve estimate: the int4 serving levers are priced; their host heap is handed back

- **`ServeSetup.exp_int4`** (`E4B_SERVE_EXP_INT4=1`, all-VRAM only).
  - The int4-b32 expert stores replace the NF4 stacks, which are freed. They hold the same bytes per weight (0.5625)
    plus each projection's fp32 split-K partials for `top_k` rows.
  - The repack's host peak is priced at `INT4_REPACK_HOST_BYTES_PER_PARAM` (12) per parameter of the largest layer: its
    fp32 reads, their fused copy and the packed lists.
  - On the device, one layer's int4 store sits beside the NF4 stacks before the KV pool is built. That is priced where
    it exceeds the serving total.
  - The source checkpoint on local disk is listed as not modelled.
  - Refused with the solver placement (`enable_serve_experts_int4` refuses tiered layers) and for gpt-oss, whose MXFP4
    store is not priced.
- **`ServeSetup.attn_int4`** (`E4B_SERVE_ATTN_INT4=1`).
  - The attention projections on the int4-b32 grid.
  - The bf16 copy each `Int4Linear` builds at its first call over 16 rows (any prefill chunk) and keeps.
  - Its preallocated GEMV and K16 workspaces.
  - Net: more memory than bf16 attention once a prompt is served. It is a decode-speed lever, not a memory one.
- **Measured** on an RTX A2000: OLMoE-1B-7B, all-VRAM, 4 sequences × 4,096 tokens, 1,024-token prompts.
  - Allocator peak against the NF4 build, priced / measured:
    - `exp_int4`: +24 MiB / +8 MiB. The load peak is +24 MiB exactly; the int4 prefill route's transients are 16 MiB
      smaller than NF4's.
    - `attn_int4`: +184.1 MiB / +184.1 MiB.
  - The repack's anonymous host peak: 10.4–11.2 B per parameter of a layer.
- **The int4 levers hand their freed host heap back** (`engines.host_heap.release_freed_host_heap`, glibc
  `malloc_trim(0)`).
  - The expert repack reads every projection in fp32 on the host, and the attention swap copies every projection to
    the host in fp32. glibc kept those freed 8-16 MiB blocks for the life of the process.
  - Measured on the run above:
    - +3.6 GB of anonymous host memory after load with `exp_int4`, +2.3 GB with `attn_int4`;
    - one trim returned 3.9 GB with both.
  - Now:
    - the repack drops each layer's fp32 stacks before the next layer's read, and trims after every layer;
    - the attention swap trims after every projection.
  - Both levers now end their load below the NF4 build's anonymous memory (0.57–0.61 GB against 0.70 GB). The
    attention swap's host peak is the NF4 build's (1.33 GB, was 3.24 GB), and the repack's fell from 7.58 GB to
    4.88 GB.
- `ServeSetup.to_env()` now always sets both lever variables (and `E4B_INT4_KEEP_NF4=0` with `exp_int4`), so an
  inherited environment cannot turn a lever on behind the estimate.
- `MoETopology.int4_attention_linears`: the `(out, in)` shape of every projection the swap would take. It comes from
  `engines.int4_attn.attention_linears`, the rule `enable_serve_attn_int4` iterates.
