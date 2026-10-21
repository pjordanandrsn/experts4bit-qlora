# p143: re-deriving its committed verdicts

Each line below rebuilds a committed verdict from committed files only, byte for byte.
`tests/test_public_rederivation.py` runs every line (CONTRIBUTING.md, "Re-derive lines"). The receipts
stay as committed: the line for an already-committed receipt lives here, never in the receipt.

- `p143-5090-2`:

  Re-derive: `python bench/p143/p143_reduce.py --bench bench/p143/receipts/p143-5090-2/p143.json --out {out} --e4b-sha adea732102bd17f0cf229d42e92449562ed0bdf6` writes `bench/p143/receipts/p143-5090-2/verdict_p143.json`

- `p143a2-5090-1`:

  Re-derive: `python bench/p143/p143a2_reduce.py --dir bench/p143/receipts/p143a2-5090-1 --out {out} --e4b-sha 7ed1695eceab0052078d43c14aed8da06381d8f6` writes `bench/p143/receipts/p143a2-5090-1/verdict_p143a2.json`

- `p143a3-5090-1`:

  Re-derive: `python bench/p143/p143a3_reduce.py --dir bench/p143/receipts/p143a3-5090-1 --out {out} --e4b-sha e8de1799769065283554cda4525be614b39d1ee3` writes `bench/p143/receipts/p143a3-5090-1/verdict_p143a3.json`
