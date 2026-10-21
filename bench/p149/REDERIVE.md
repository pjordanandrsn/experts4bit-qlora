# p149: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `p149-5090-2`:

  Re-derive: `python bench/p149/p149_reduce.py --dir bench/p149/receipts/p149-5090-2 --out {out} --e4b-sha 835874d1449314fc569442228a3e46e7a6602748` writes `bench/p149/receipts/p149-5090-2/verdict_p149.json`
