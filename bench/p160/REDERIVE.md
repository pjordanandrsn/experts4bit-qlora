# p160: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `p160-5090-2`:

  Re-derive: `python bench/p160/p160_reduce.py --bench bench/p160/receipts/p160-5090-2/bench_p160.json --out {out}` writes `bench/p160/receipts/p160-5090-2/verdict_p160.json`
