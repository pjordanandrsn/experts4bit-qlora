### P149 box K's results (K_GO) and `E4B_SERVE_ATTN_INT8`'s status line

`bench/p149/RESULTS-p149.md` records box K, `p149k-5090-8`: K37's int8 attention projections against cuBLAS bf16 at
Mixtral-8x7B's shapes, a 0.984 ms W1 projected in-situ saving against the 0.1625 ms bar, read as K_GO. That is a
microbench premise, never a served gain. The run's public receipt is under `bench/p149/receipts/p149k-5090-8/`, with
the files kept only in the private receipt store named by sha256. `docs/STATUS.md` now says `E4B_SERVE_ATTN_INT8=1`
(#1848) is opt-in on Mixtral, with no speed claimed until P149's box Q+S reads.
