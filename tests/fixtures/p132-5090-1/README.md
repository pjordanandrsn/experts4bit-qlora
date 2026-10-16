# p132-5090-1: the in-box canary's records (strike 1)

These are the four canary records of lane P132's first box run, `p132-5090-1` (2026-10-11, RTX 5090, HARNESS_ERROR at the
canary, rc 26). They are copied byte for byte from the receipt store, commit `814abd9e`
(`receipts/experts4bit-qlora/2026-10-11/p132-5090-1/p132/`). `SHA256SUMS` holds the store's bytes.

- `canary.json`: the canary driver's record (buckets 1..4 captured, one replay per condition, then the census verdict).
- `canary_drv_n2_graph.json`, `canary_drv_n2_node.json`: the two short census drivers (8 replays bracketed each).
- `canary_census.json`: SC1b's reading of the two short captures.
  - Both modes void: "5 steady steps kept < 6" and "5 steady launches kept < 6".
  - The nsys sqlite exports were not preserved, so `tests/test_p132.py` rebuilds exports of the same shape with SC1b's
    own `_mkdb`.
