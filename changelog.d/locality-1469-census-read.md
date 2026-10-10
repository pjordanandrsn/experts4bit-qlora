### #1469 item 1 census read: expert locality on Qwen3-30B-A3B (bench, receipts and tests only)

- `loc-a4000-1` (one RTX A4000, $0.241) recorded every routed expert id: decode is four prompt kinds × 512 greedy
  steps, and prefill is 16 wikitext windows. The receipts are in `bench/locality-1469/receipts/loc-a4000-1/`.
- Distinct experts per layer in a window, mean over layers: decode 42.4 / 53.5 / 63.9 / 73.0 at W = 16 / 32 / 64 /
  128; prefill 39.0 / 50.8 / 61.6 / 71.9. Churn is 0.612 for decode and 0.519 for prefill. The hot set of 8 per
  layer serves 12.9 % of decode's routed slots.
- `tests/test_locality_1469_read.py` checks the npz format and recomputes both committed summaries from the npz.
