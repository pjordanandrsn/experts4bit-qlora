# p161: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `p161-5090-1`:

  Re-derive: `python bench/p161/p161_reduce.py --dir bench/p161/receipts/p161-5090-1 --out {out} --e4b-sha da3eb5c5e452e0031659e2e219b63bb130e79e3b` writes `bench/p161/receipts/p161-5090-1/verdict_p161.json`
