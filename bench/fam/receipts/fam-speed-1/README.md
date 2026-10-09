# `fam-speed-1`: the router epilogue's speed on Qwen3.6-35B-A3B at one row (Amendment 4's reading)

One RTX 5090 (cc 12.0, driver 580.126.09) on an AMD EPYC 7713 host (128 CPUs): Vast instance 54968853, launched
2026-10-09T05:08:24Z, torn down 05:20:40Z. Cost $0.632. Code: e4b `48cfccaa` (Amendment 4's merge, #1436),
grouped-nf4-gemm 0.43.0 (`6ee2e10`).

**Verdict: FASTER.** The median one-row decode step was 12.643 ms OFF and 12.298 ms ON in block a, and 12.627 and
12.289 ms in block b. The ratios are 0.9727 and 0.9732, both under 0.98, and the blocks agree within 0.06 %.
- **The instrument held.** The census read `0 / 0 / [0, 0] / 40`, and each setting swapped 40 routers. Every runner
  captured buckets 1–16 and replayed bucket 1 exactly 293 times, with no eager step. Each setting emitted the same
  tokens in both blocks.
- **Tokens:** ON and OFF agree on 293 of 293 in both blocks (reported, never gated).
- **Reported:** per-pair ratio medians 0.9728 and 0.9732; SM clock 2542–2917 MHz; peak 23.1 GiB allocated
  (23,657 MiB).
- **The default install:** the image has no `causal_conv1d` or `flash-linear-attention`, so Qwen3.6's linear layers run
  transformers' reference path in both settings. This read is the default server as e4b installs it.

The maintainer re-derived the verdict from the store (`bc74955d`) byte for byte. The launcher's receipt and ledger row
are in the receipt store. `SHA256SUMS` covers every file here.
