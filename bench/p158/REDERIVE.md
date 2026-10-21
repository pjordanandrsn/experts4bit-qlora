# p158: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `p158-5090-1`:

  Re-derive: `python bench/p158/p158_reduce.py --dir bench/p158/receipts/p158-5090-1 --out {out} --e4b-sha 406a391a11d42ddb39154cfc36938a2da5d68216` writes `bench/p158/receipts/p158-5090-1/verdict_p158.json`
