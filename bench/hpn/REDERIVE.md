# hpn: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `hpn-a2000-read8`:

  Re-derive: `python bench/hpn/hpn_reduce.py bench/hpn/receipts/hpn-a2000-read8/hpn_spec.public.json --out {out}` writes `bench/hpn/receipts/hpn-a2000-read8/hpn/result.json`
