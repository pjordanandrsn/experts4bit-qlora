### The paged server serves a merged arena with its own adapter, opt-in (refusal 13)

- `E4B_PAGED_MERGED_ARENA=1` serves a merged arena (`experts4bit_qlora.merged_arena`) with the adapter it was merged
  from. It is off by default and stays opt-in until the registered served read gates it (bench/d0/ARENA-DESIGN.md).
- **Refusal 13** is decided before the load, from records only, for every config. It refuses a merged arena:
  - without the opt-in;
  - without an adapter, which would serve a trained function under the base's name;
  - with an adapter other than the one it was merged from: its id, expert tensors, recorded base or scale.

  The opt-in without a merged arena is a loose flag.
- When refusal 13 passes, a live expert LoRA is not refusal 1. The delta is already in the arena's rows, so the server
  applies none of the adapter's expert tensors; the attention LoRA merges once at load, as for any adapter.
- After the load, the served experts are compared with the merged arena's own fingerprint (`merge.merged_qfp_v1`). A
  mismatch is refusal 12, with no override.
- Only the arena's own verified `merge` block lets the loader (`merged_arena=`) and the hybrid tier (`merged_arena=`)
  read a merged arena. `/health` carries the block, and `base_match.experts` reads `merged (...)`.
- `docs/ADAPTER_PORTABILITY.md`'s expert matrix names the path and its gate.
