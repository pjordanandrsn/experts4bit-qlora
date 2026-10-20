### P158 read (#1313): K39's decode in the served step, FLIP (g16 1.0925)

On the shipped default server for Qwen3-30B-A3B, `E4B_K25_LUT=prmt` (K39's decode for K25, grouped-nf4-gemm v0.50.0)
decoded 16 concurrent requests 9.2 % faster than the select tree: g16, the min of four ABBA pairs, 1.092458 against the
1.03 bar; one request unchanged (g1 0.999856 against 0.99). Every token matched the tree's, the tree's own null held
across its five processes, and every build recorded exactly 960 K25 calls and 480 GEMV calls. All six predictions held.
One RTX 5090, $0.419. `bench/p158/RESULTS-p158.md` and `bench/p158/receipts/p158-5090-1/`. A default flip for
qwen3_moe follows as its own PR.
