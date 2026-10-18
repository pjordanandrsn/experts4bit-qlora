# `p141-5090-2`: K33 at Mixtral-8x7B's expert shapes, the reading

One RTX 5090 (cc 12.0, 170 SMs, driver 595.71.05) on an Intel Xeon Gold 6240 host (36 CPUs): Vast instance 56580839,
launched 2026-10-18T16:15:06Z, torn down 17:13:24Z. Cost $0.999. Code: e4b `d2bf9f64` (the registration's merge, #1754),
grouped-nf4-gemm 0.46.1 (`70bc03cb`); torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2.

**Verdict: LICENSED.** At one request the default server (16 x 768, graphs on) stepped in 19.535 and 19.624 ms with the
scalar GEMV (B0a, B0b) and in 8.114 and 8.126 ms with K33 at phase 1's plans (B1a, B1b): g1 = min(2.4075, 2.4148) =
**2.4075** (bar 1.03). At 16 requests: 53.16 / 53.28 ms against 53.35 / 53.31 ms, g16 = **0.9964** (no-regression line
0.99).
- **Phase 1** (`select.json`): all twelve of K33's plans met the contract at both shapes. The lowest medians chose
  `28672,4096=16,1024,4,1` and `4096,14336=16,512,4,2`. Reported, never ruled: 3.09 ms against the scalar GEMV's 10.31
  ms for gate_up's 32 layers, and 1.95 against 6.10 ms for down's; the copy floor is 2.82 and 1.41 ms.
- **The instrument held.** The premise passed on the card (kernel contract 35, served route 18, none skipped). The fit
  check passed at 16 x 768. Every bucket was captured. Engagement was exact: B0 dispatched only the scalar GEMV and its
  split-K form, and B1 only `bw_prmt32`, with `bw_splitk` beside it at down's split-K plan. The self-pairs read 0.9955
  and 0.9985 at W1 and 0.9978 and 1.0009 at W16.
- **Quality** (P110's bar, 24 windows per text): wikitext bias +0.00228 (bar 0.01165), spread 0.00826 (bar 0.01244), K8
  Δppl +0.0103 (≤ 0.05); c4val1 bias −0.00096 (bar 0.01038).
- **Tokens** (reported): B1's equal B0's on the W1 row at both lengths, and on 15 of 16 W16 rows at 160 tokens.

The reviewers and the maintainer each re-derived the verdict from the receipt store (`279938dd`). The launcher's receipt
and ledger row are in that store. `SHA256SUMS` covers every file here.
