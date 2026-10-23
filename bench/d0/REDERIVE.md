# D0: re-deriving the result from the public record

The reducer that wrote each fixture's summary is in the receipt, byte for byte. It needs only the Python standard library.
From the repository root:

    python bench/d0/receipts/d0-p224s3/tools/d0_reduce.py --pass-dir bench/d0/receipts/d0-p224s3/dq/pass --out dq.json
    python bench/d0/receipts/d0-p224s3/tools/d0_reduce.py --pass-dir bench/d0/receipts/d0-p224s3/nodq/pass --out nodq.json

Each output equals the receipt's `summary.json` for that fixture (`tests/test_d0_results.py` checks it). The bootstrap is
seeded (2,000 resamples, seed 0).

Re-running the cells themselves needs the fixture tensors, which are not in this repository, and the base checkpoint
OLMoE-1B-7B-0924-Instruct at `7f1c97f4`. The receipt's README names them.
