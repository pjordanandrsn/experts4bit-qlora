### D0 follow-on design note: a merged, re-quantized expert arena (USD 0, no code)

- `bench/d0/ARENA-DESIGN.md`: how path (a) bakes a trained expert delta into a new NF4 arena offline. It covers a
  byte-for-byte parity test against the real bake plus a self-check on every base row, the merge source D0 measured, the
  provenance it records (the merged arena's own fingerprint included), and the server rules, grouped by phase, that keep
  serving it opt-in and fail-closed until the registered served read licenses it.
