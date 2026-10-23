# D0: re-deriving the result from the public record

The reducer that wrote each fixture's summary is in the receipt, byte for byte. It needs only the Python standard library.
From the repository root:

    python bench/d0/receipts/d0-p224s3/tools/d0_reduce.py --pass-dir bench/d0/receipts/d0-p224s3/dq/pass --out dq.json
    python bench/d0/receipts/d0-p224s3/tools/d0_reduce.py --pass-dir bench/d0/receipts/d0-p224s3/nodq/pass --out nodq.json

Under Python 3.11, the version that wrote them, each output is byte-identical to the receipt's `summary.json` for that
fixture. Under other versions the last digits of a float can differ (`sum()` changed in 3.12); no label, resolution or
verdict moves. `tests/test_d0_results.py` checks the floats to a relative 1e-9. The bootstrap is seeded (2,000
resamples, seed 0).

Re-running the cells themselves needs the fixture tensors, which are not in this repository, and the base checkpoint
OLMoE-1B-7B-0924-Instruct at `7f1c97f4`. The receipt's README names them.
