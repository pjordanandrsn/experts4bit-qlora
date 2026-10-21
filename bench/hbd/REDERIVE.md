# hbd: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `hbd-5090-1`:

  Re-derive: `python bench/hbd/hbd_reduce.py bench/hbd/receipts/hbd-5090-1/hbd_spec.public.json --out {out}` writes `bench/hbd/receipts/hbd-5090-1/hbd/result.json`
