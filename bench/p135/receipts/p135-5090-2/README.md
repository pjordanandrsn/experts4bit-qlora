# `p135-5090-2`: what Mixtral's ON_auto folds buy at one row (the reading)

One RTX 5090 (cc 12.0, driver 580.95.05) on an AMD EPYC 7742 host (128 CPUs): Vast instance redacted-6331693d, launched
2026-10-18T11:54:56Z, torn down 12:20:22Z. Cost $1.159. Code: e4b `5be4c4d9` (Amendment 1's merge, #1737),
grouped-nf4-gemm 0.45.0 (`724ccc45`); torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0.

**Verdict: FASTER.** The median one-row decode step was 20.244 ms OFF and 19.016 ms ON in block a, and 20.515 and
19.031 ms in block b. The ratios are 0.9393 and 0.9277, both under 0.98, and the blocks agree within 1.26 % (the NOISY
line is 1.5 %). The folds save 1.23 and 1.48 ms a step.
- **The instrument held.** The census read `0 / 65 / [32, 32] / 32`, and each setting swapped all 161 fold-patched
  forwards (65 / 64 / 32 by module). The router epilogue patched 32 routers, all on their fp32 upstream path. Every
  runner captured bucket 1 and replayed it exactly 293 times, with no eager step. Each setting emitted the same tokens
  in both blocks.
- **The premise:** 35 passed, none skipped, on the card (`logs/premise.log`).
- **Tokens:** ON and OFF agree on 283 of 293 in both blocks and first part at step 38 (reported, never gated; the
  folds change arithmetic, which FAM's quality read licensed).
- **Reported:** per-pair ratio medians 0.9391 and 0.9293; SM clock 2910–2955 MHz; peak 27.3 GiB reserved
  (27,952 MiB).
- **Predictions:** Y1–Y6 all held (`../../PREREG-p135.md`).

The reviewers and the maintainer each re-derived the verdict from the store (`1342d2db`) byte for byte. The launcher's
receipt and ledger row are in the receipt store. `SHA256SUMS` covers every file here.
