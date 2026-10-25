# `p143-5090-1`: step 1's first attempt (HARNESS_ERROR, strike 1)

One RTX 5090 (cc 12.0, 170 SMs, driver 575.64.03, a 400 W power limit) on an AMD Ryzen Threadripper PRO 7975WX host:
Vast instance redacted-fd0c28b5 on machine 40521, 2026-10-18T17:18:12Z to 17:20:19Z. Cost $0.008. Code: e4b `c246617a` (the
registration's merge, #1759), grouped-nf4-gemm 0.46.1 (`70bc03cb`).

The refusals, install, tripwire and premise passed (19 passed). `p143_bench.py` then stopped before timing any arm.
`Fp8PagedKV.rewind` refuses a forward move (`rewind forward (layer 0: 256 -> 640)`), and the bench walked its contexts
ascending after filling slot 0 to 4096 tokens. No arm was timed and no data was seen. This was the lane's strike 1.
Amendment 1 (#1762) takes the contexts longest first and runs the same setup on CPU in CI.

The launcher's receipt and ledger row are in the receipt store (`a1d5a12e`). `SHA256SUMS` covers every file here.
