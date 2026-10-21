# p150: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `p150-5090-1`:

  Re-derive: `PYTHONPATH=bench/p125 python bench/p150/p150_reduce.py --dir bench/p150/receipts/p150-5090-1 --out {out}` writes `bench/p150/receipts/p150-5090-1/verdict_p150.json`
