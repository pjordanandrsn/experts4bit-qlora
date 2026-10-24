### Changed

- Eight receipt directories merged on 2026-10-23 (p223-5090-1, p225-5090-r3, p226-5090-1, p227-5090-1, p228-5090-r1,
  p229-5090-3, p230-5090-r1, p232-5090-1) no longer carry the rented card's GPU UUID. `forensics.txt` and `outer.log`
  are store-only (`STORE-ONLY.sha256`, byte-exact in the private receipt store), as the receipts rule has required since
  2026-10-20. p225's trainer receipt is replaced by a public projection of the three fields `p225_gate.py` reads, and
  the gate's check passes the same on it. Every `SHA256SUMS` still verifies.
- The private-marker guard adds an added-lines-only rule for GPU UUIDs, with its canary line, so a new receipt cannot
  add one. Files merged before the rule keep theirs.
