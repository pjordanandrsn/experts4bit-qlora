### Receipts: rented-hardware identifiers replaced by keyed tokens across the public tree

Public receipts no longer carry the rented card's GPU UUID or the provider's instance id
(`docs/receipts-redaction.md`). 2155 files changed in place, forward only. Git history is untouched, and the private
receipt store keeps every original byte-exact.
- **Tokens, not blanks:** each distinct value became one keyed token (`GPU-redacted-<8 hex>`,
  `instance-redacted-<8 hex>`), so `ra_reduce`'s run identity still tells cards apart.
- **Checkable:** `docs/receipts-redaction.sha256` lists each file's original and redacted sha256. Receipts with a
  `SHA256SUMS` are re-pinned and name their store-held originals under `# same bytes` lines.
  `scripts/redact_receipt_ids.py --verify` re-derives every file from v0.58.1 given the key.
- **Left as they were:** 44 files keep their identifiers. They are sealed or hash-pinned, read by P132's card check, or
  tests that feed synthetic values. `tests/test_receipts_redaction.py` pins the list.
- **Guard:** `.github/private-markers.txt` now refuses the GPU-UUID shape across the whole tree, and an instance id in
  added lines.
