# `fam-qw36-1`: Qwen3.6-35B-A3B, the B=1 fused stack at T == 1 (VOID: a harness bug)

One RTX 5090 (cc 12.0, driver 595.91.07) on an AMD EPYC 7763 host (256 CPUs): Vast instance 54930776, launched
2026-10-08T23:33:38Z, torn down 2026-10-09T00:37:40Z. Cost $1.639. Code: e4b `9766fb4c`, grouped-nf4-gemm 0.43.0
(`6ee2e10`).

**Verdict: VOID** ("OFF: record missing or not ok", the same for ON_auto). The premise passed (36 tests), and the
fetch and bake completed. The OFF process scored wikitext at shape 1 (sets A, B and C), then died building the first
shape-12 runner (`logs/fam_qw36_OFF.log`): e4b's `LinearStatePool.ensure_slots` refused to grow the hybrid's
linear-state pool from 17 slots to 28 under a frozen decode-graph setup. The box ran shape 1 first, so its first
runner sized the pool. No record was written, and ON_auto did not run.

Amendment 2 runs the largest shape first, and the registration's one rerun is `fam-qw36-2`. The launcher's receipt and
ledger row are in the receipt store (`3b11414c`). `SHA256SUMS` covers every file here.
