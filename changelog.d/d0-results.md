### D0 results: a trained expert-LoRA delta survives repacking (one OLMoE-1B-7B fixture)

- `bench/d0/RESULTS-d0.md` reports D0, run on CPU under its registration and re-derived from the receipt store.
  - The expert delta's held-out improvement is RESOLVED at about 11 SE.
  - All 12 cells (NF4, INT4 and MXFP4, from the trained base or raw W, on two fixtures) are RETAINED. The merged-and-repacked
    model sits within −0.0036 to +0.0167 nats of the unmerged adapter.
  - The registration's prediction that NF4 from the trained base would lose the delta is MISSED.
  - The scope is one fixture.
- The receipt is under `bench/d0/receipts/d0-p224s3/`, including the tools as run. `bench/d0/REDERIVE.md` and
  `tests/test_d0_results.py` re-derive both summaries from it.
