### Fixed

- `tests/test_receipt_sums.py`: every line of a committed receipt's `SHA256SUMS` must now be a file git tracks, or a
  line of that receipt's `STORE-ONLY.sha256` at the same sha256. It reads git's index, so a log that exists on disk
  but is ignored (`.gitignore` has `*.log`) no longer passes. The frozen `LEGACY_GAPS` table is gone: the 18 older
  receipts it listed are complete. 34 of their logs are committed byte-for-byte from the private receipt store, and
  15 new `STORE-ONLY.sha256` files name the other 149 files, each grouped under a header that says where the bytes are
  and, where it is recorded, why. No existing receipt file is changed.
