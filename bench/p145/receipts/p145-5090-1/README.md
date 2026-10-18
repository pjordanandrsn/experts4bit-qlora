# `p145-5090-1`: Mixtral-8x7B's one-request step after K33, kernel by kernel (the read)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.71.05) on an Intel Core Ultra 9 285K host (24 CPUs, 188 GB RAM): Vast instance
56617171, launched 2026-10-18T20:56Z, torn down 21:12:02Z. Cost $0.816. Code: e4b `67ba40b7` (the registration's merge,
#1771), grouped-nf4-gemm 0.47.0 (`eea5df6b`); torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2,
Nsight Systems 2025.6.1.190.

**Verdict: READ** (`verdict_p145.json`). No VOID condition fired:
- the self-tests and P136's 19 premise tests passed;
- both batches' per-name tables close every replay exactly, with 61 kept steps each;
- the W1 expert path ran 32 launches per step for each of gate_up, `_swiglu_rows` and down.

The unprofiled arms stepped in 8.2499 and 8.2559 ms at one request (self-pair 0.9993) and 53.543 and 53.490 ms at sixteen.
The W1 node replay period is 8.125 ms. `RESULTS-p145.md` has the ranking, the bandwidth floors and the proposal.

The maintainer re-derived `verdict_p145.json` byte-identical from the receipt store (`f3ce417a`). The launcher's receipt
and ledger row are in that store. The raw Nsight exports and the checkpoint stayed on the box. `SHA256SUMS` covers every
file here. Thirteen logs with CRLF line endings stay only in that store,
byte-exact; `STORE-ONLY.sha256` lists them.
