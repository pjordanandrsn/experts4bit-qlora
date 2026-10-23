### D0 follow-on design note: a merged, re-quantized expert arena (USD 0, no code)

- `bench/d0/ARENA-DESIGN.md`: how path (a) bakes a trained expert delta into a new NF4 arena offline, with a byte-for-byte
  parity test against the bake, the merge source D0 measured, the provenance it records, and the server rules that keep
  serving it opt-in and fail-closed until the registered served read licenses it.
