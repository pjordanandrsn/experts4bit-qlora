### TC1 amendment 79 read: NOT_READ (e4b 0.58.0 against Unsloth 2026.10.8)

- One reading box, `tc1-5090-pos79-r1` on an RTX 5090: all four profiled arms ran 60 steps at the release pins, and Unsloth's wheels were
  proven byte-identical to PyPI's. A first box was destroyed at pre-flight (still loading after 600 s) before any command ran. Lane
  spend USD 2.123, against USD 5.93 registered worst.
- NOT_READ. Both e4b draws are VALID. Both Unsloth draws are VOID on TC1's standing engagement predicate, which counts
  `torch._grouped_mm` calls and read 0. So no position, ratio or equivalence reading is quoted, e4b's own draws included.
- The receipts show Unsloth's own backend counters non-zero, and each Unsloth profile's op list names `unsloth_zoo::grouped_mm_triton`
  with no `aten::_grouped_mm`. The hypothesis that Unsloth 2026.10.8's `grouped_mm` backend dispatches to its own Triton op is labelled
  UNTESTED. The predicate change is for the next release.
- `bench/h2h-2026-10-02/tc1/RESULTS-tc1-pos79.md` holds the note. `receipts/tc1-5090-pos79-r1/` holds the public copy: 35 files
  byte-identical, 26 store-only, and the derived `public/`, which rebuilds the box's RESULTS byte for byte through `tc1_public.py`.
