# p156: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `p156-5090-1`:

  Re-derive: `python bench/p156/p156_reduce.py --bench bench/p156/receipts/p156-5090-1/bench_p156.json --out {out}` writes `bench/p156/receipts/p156-5090-1/verdict_p156.json`
