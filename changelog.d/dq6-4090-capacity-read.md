### DQ6 read: on a 24 GB RTX 4090, streaming trains 4.75× the resident sequence length (CAP_REAL; bench, receipts and docs; #1083)

- **The read.** DQ4's capacity read on one 24 GB RTX 4090 (`dq6-4090-1`, $0.802), with the chunked loss and the
  default allocator: resident L\* 2048 tokens, streamed 9728, so **G = 4.75**, bracket [3.80, 5.00]. With
  `expandable_segments` it reads 5.20 (2560 → 13312).
- **Against the 32 GB card.** DQ4's 5090 read 2.00×. The resident weights take most of a 24 GB card, while the
  streamed model keeps two layers of them, so the factor more than doubles.
- **How it checks out:**
  - every fresh confirmation agreed with its ladder;
  - R − S allocated is 14.07–14.08 GiB at every common rung, which is the weights less two layers;
  - step time is unchanged at the registered rung (T(S)/T(R) 0.993 at 2048 tokens).
- **What is new in the tree:**
  - `bench/dq6/RESULTS-dq6.md`;
  - the receipts, re-derived byte for byte by a new lane test;
  - a two-card capacity table in `docs/CHOOSING.md`.
