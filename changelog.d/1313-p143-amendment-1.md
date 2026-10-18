### P143 Amendment 1 (#1313): the bench takes its contexts longest first, and CI runs its setup on CPU

`p143-5090-1` read HARNESS_ERROR ($0.008) before any timing. `Fp8PagedKV.rewind` refuses a forward move, and the bench
walked its contexts ascending after filling to 4096. The setup is now `fill`, `hold_args` (longest first) and
`check_held` in `bench/p143/p143_bench.py`. `tests/test_p143.py` runs that same setup on CPU, on e4b's own KV class at
the registered contexts, wherever grouped-nf4-gemm is installed. The grid, the timing, the rule and the box terms are
unchanged. The rerun is `p143-5090-2`; `-3` is its pre-create spare.
