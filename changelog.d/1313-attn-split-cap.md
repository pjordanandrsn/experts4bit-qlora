### `E4B_ATTN_SPLIT_CAP` (#1313, lane P143 Amendment 2): cap the paged FP8 attention decode's splits (off by default)

`E4B_ATTN_SPLIT_CAP=n` hands the paged FP8 decode attention `split_cap=n` (grouped-nf4-gemm#558). The kernel's own
planner, `decode_split_plan`, then runs `min(n, its plan)` splits. That plan sizes splits from the block table's
capacity: 32 at B=1 for Qwen3-30B-A3B's 4 KV heads on an RTX 5090. P143's microbench timed 16 splits at 7.6 µs per
launch against 11.2 at 640 positions, and 14.4 against 15.3 at 4096. e4b computes no plan.
The cap is set on the KV before any decode graph is captured. It is handed over only on the fp8 split path, only when
the installed kernel takes `split_cap`, and never over an explicit `n_split`. Unset resolves per family; no family takes
it until P143's served read. `0` keeps the library's plan. The build's `info.attn_split_cap` records the cap, its source
and what each captured bucket planned.
