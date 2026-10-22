# P222: re-deriving the verdict from the public record

The registered reducer reads the bench record and writes the verdict. From the repository root:

    python bench/p222/p222_reduce.py --bench bench/p222/receipts/p222-5090-5/bench_p222.json --out verdict.json

Under Python 3.11 the output is byte-identical to `receipts/p222-5090-5/verdict_p222.json`
(`tests/test_p222.py` checks it). Under other Pythons the last digit of a standard error can differ; no value that
decides the verdict moves.
