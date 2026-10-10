# `fam-mixtral-prove-1`: Mixtral's own proof, first attempt (NOT PROVED: out of GPU memory)

One RTX 5090 (cc 12.0, driver 595.71.05) on an Intel Xeon E5-2699 v3 host (72 CPUs): Vast instance 55040837,
launched 2026-10-09T14:51:19Z, torn down 15:31:22Z. Cost $1.115. Code: e4b `f0f8e2d7` (Amendment 5's merge, #1457),
grouped-nf4-gemm 0.43.0 (`6ee2e10`).

**Verdict: NOT PROVED** (every record missing). The premise (36 tests) passed, the 93 GB fetch took 1986 s, and the
bake was OK. The OFF process then ran out of GPU memory building the server's own KV pool: the default 16 slots × 4096
tokens did not fit beside about 26.8 GiB of NF4 weights (`logs/fam_mixtral_OFF.log`). The instrument never uses that
pool. Amendment 7 builds it at 768 tokens a slot.

The launcher's receipt and ledger row are in the receipt store (`bfc1cd3b`). `SHA256SUMS` covers every file here.
