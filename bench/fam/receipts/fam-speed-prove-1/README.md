# `fam-speed-prove-1`: Amendment 4's speed proof on Qwen3.6-35B-A3B (not a reading)

One RTX 5090 (cc 12.0, driver 580.126.09) on an AMD EPYC 7713 host (128 CPUs): Vast instance 54967271, launched
2026-10-09T04:51:23Z, torn down 05:03:41Z. Cost $0.632. Code: e4b `48cfccaa` (#1436's merge), grouped-nf4-gemm 0.43.0
(`6ee2e10`). The speed box end to end at 32 timed steps a setting.

**Verdict: PROVED (rc 0).** The speed reducer did not VOID.
- The census was `0 / 0 / [0, 0] / 40`, and 40 routers were swapped in each setting.
- Every runner captured buckets 1–16 and replayed bucket 1 exactly 69 times, with no eager step.
- Each setting emitted the same tokens in both blocks.
- At proof scale it read FASTER (0.9722 and 0.9733); that is not a reading. Peak 23.1 GiB allocated (23,648 MiB).

The maintainer re-derived the proof from the store (`ed46623d`). The launcher's receipt and ledger row are in the
receipt store. `SHA256SUMS` covers every file here.
