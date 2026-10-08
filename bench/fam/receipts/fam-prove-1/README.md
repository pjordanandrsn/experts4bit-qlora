# `fam-prove-1`: the proving rental (not a reading)

One RTX 5090 (cc 12.0, driver 595.71.05) on an AMD EPYC 7C13 host: Vast instance 54917126, launched
2026-10-08T21:48:57Z, lane done 22:17:44Z, torn down 22:17:45Z. Cost $0.423. Code: e4b `9629874f` (the #1380 merge),
grouped-nf4-gemm 0.43.0 (`6ee2e10`), torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0. Granite-3.1-3b-a800m at
32 teacher-forced positions, every cell.

**Verdict: VOID on census, so the proof failed (rc 27), as `bench/fam/PREREG-fam.md` registers.**
- The router epilogue licensed 27 of Granite's 32 routers (`failed_probe: 5` in `logs/fam_granite_ON_epi.log` and
  `logs/fam_granite_ON_auto.log`). The seat A2000 had licensed 32 at the same code and weights.
- #1398 is the fix: the probe now judges decisive rows on fp32 CPU logits over 64 rows.
- Every other integrity check held: OFF's census zero, no int4 store, decode-attention calls, decode-shaped forwards,
  wrapper coverage, windows, prefill sizes and group sizes.
- The premise passed (36 tests), and so did all three self-tests.
- The anchor's two processes were skipped by the time-left rule. That, and this run's step time (49.0 ms median eager at
  T == 1), sized Amendment 1.

The launcher's receipt and ledger row are in the receipt store. `SHA256SUMS` covers every file here.
