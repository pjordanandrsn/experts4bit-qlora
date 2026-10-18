### `E4B_ATTN_SPLIT_CAP` (#1313, lane P143 Amendment 2): cap the paged FP8 attention decode's splits (off by default)

`E4B_ATTN_SPLIT_CAP=n` makes the paged FP8 decode attention run `min(n, the library's own plan)` splits. The plan sizes
splits from the block table's capacity: 32 at B=1 for Qwen3-30B-A3B's 4 KV heads on an RTX 5090. P143's microbench timed
16 splits at 7.6 µs per launch against 11.2 at 640 positions, and 14.4 against 15.3 at 4096. The cap is set on the KV
before any decode graph is captured, applies only on the fp8 split path, and never overrides an explicit `n_split`.
Unset resolves per family; no family takes it until P143's served read. `0` keeps the library's plan.
`fp8_paged_kv.gnf4_split_plan` copies grouped-nf4-gemm's inline plan. `tests/test_attn_split_cap.py` pins the installed
block's sha256 and runs that block against the copy over a grid. A public `plan_n_split` (grouped-nf4-gemm 0.48) will
replace both.
