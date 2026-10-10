# `fam-mixtral-3`: Mixtral's reading, first attempt (VOID: a serving regression on main)

One RTX 5090 (cc 12.0, driver 595.84) on an AMD EPYC 9454 host (96 CPUs): Vast instance 55076206, launched
2026-10-09T18:53:29Z, torn down 19:15:17Z. Cost $0.87. Code: e4b `4b9954f5` (Amendment 8's merge, #1479),
grouped-nf4-gemm 0.43.0 (`6ee2e10`).

**Verdict: VOID** (every record missing). The premise (36 tests), the self-tests, the fetch and the bake passed. The
OFF process then raised `TypeError: _HybridTier.forward() got an unexpected keyword argument 'residual'` on its first
forward (`logs/fam_mixtral_OFF.log`). #1477, merged after Amendment 8's review, handed `residual=` to the hybrid tier
that the default server installs, which did not take it. #1482 fixed it, and Amendment 9 moved the reading past it.

The launcher's receipt and ledger row are in the receipt store (`97d853b6`). `SHA256SUMS` covers every file here.
