### serve_paged: `E4B_PAGED_MAX_SEQS=auto` by default, sized by the serve estimate (lane SC2e; `16` restores the old default)

- **The licence.** Lane SC2e (#846, read `bench/h2h-2026-10-02/sc2e/`) read `SLOTS_LICENSED(64, auto)` on Qwen3-30B-A3B
  int4 on one RTX 5090. 64 slots on the default bucket list held the SLO to 8 req/s against 4 at 16 slots, with serial
  output byte-identical and serial TTFT / TPOT within 1 %. The ruling on #1320 scoped the default to the slot count,
  estimate-sized, on the default bucket list.
- **The change.**
  - `E4B_PAGED_MAX_SEQS` unset, empty or `auto` resolves when the engine builds, before any weight is read
    (`serve_paged.resolve_max_seqs`). It takes the widest width SC2e read that fits the device's free memory: 64 or 16
    on the default bucket list; 64, 32 or 16 with `E4B_PAGED_BUCKETS=auto`. An integer keeps its meaning.
  - The fit (`serve_recipe.choose_max_seqs`): the estimate's device total for the width, a reserve for the prefill
    graph's pool (`prefill_graph_reserve_bytes`: `chunk_tokens x hidden_size x layers x 16 B`, above every pool
    measured), and 1 GiB must fit. No CUDA device, the solver placement, or nothing fitting: 16.
  - `/health` gains `engine.max_seqs_requested` and `engine.max_seqs_resolution`.
- **What it picks for Qwen3-30B-A3B int4.** 64 on an RTX 5090 at 2,048 tokens a slot; 16 at the default 4,096; 16 on a
  24 GB card.
- **The way back.** `E4B_PAGED_MAX_SEQS=16`.
- **Not changed.** The bucket list (`E4B_PAGED_BUCKETS=auto` stays opt-in until a teacher-forced read at buckets 32 and
  64), the estimate itself, routes and kernels. Speed is read on one model, one card, 512-token prompts and 2,048
  tokens a slot.
- **Tests.** The chooser on Qwen3-30B-A3B's shape: 64 on a 5090 at 2,048 tokens, never 64 at 4,096, 16 on a 24 GB
  card; a hybrid's per-slot state in the price and no prefill-graph reserve; 16 without a CUDA device or under the
  solver; the reserve against every measured pool. The env parser, the resolution's buckets, its fallback, and
  `/health`.
