### Receipts: rented-hardware identifiers replaced by keyed tokens across the public tree

Public receipts no longer carry the rented card's GPU UUID, the provider's instance id, or the box's ssh endpoint or
public address (`docs/receipts-redaction.md`). 2220 files changed in place, forward only; 7 binaries left the tree for
the private receipt store; one reproduction archive was re-packed. Git history is untouched, and the private receipt
store keeps every original byte-exact.

- **Tokens, not blanks:** each distinct value became one keyed token (`GPU-redacted-<8 hex>`,
  `instance-redacted-<8 hex>`, `ssh-redacted-<8 hex>`, `ip-redacted-<8 hex>`), so `ra_reduce`'s run identity still
  tells cards apart.
- **Driven by values, checked three ways:** the transform replaces the store's recorded values. An oracle over the
  store's records, a shape-and-context test in CI that never consults the map, and a byte guard on every change each say
  nothing was missed. The CI test found 13 instance ids and an ssh endpoint that only the public tree had recorded.
- **Checkable:** `docs/receipts-redaction.sha256` lists each file's original and redacted sha256 (`removed` for the
  binaries). Receipts with a `SHA256SUMS` are re-pinned and name their store-held originals under `# same bytes` lines.
  `scripts/redact_receipt_ids.py --verify` re-derives every file from `069a029b` given the token map.
- **dq9-5090-2's archive** was re-packed deterministically (`scripts/repack_tar.py`). Its reproduction recipe still
  passes and `dq9_reduce`'s output is byte-identical.
- **The claims register:** 390 claims' host text changed by the substitution only. The test proves every other field
  is equal and every changed text is its original with identifiers turned into tokens.
- **Left as they were:** 44 files keep their identifiers. They are sealed or hash-pinned, read by P132's card check, or
  tests that feed synthetic values. `tests/test_receipts_redaction.py` pins the list.
- **Guard:**
  - `.github/private-markers.txt` refuses the GPU-UUID shape across the whole tree.
  - The new `scripts/check_public_ids.py` reads bytes, which `git grep -I` cannot: UUIDs in binaries and gzip members,
    and in every added line any instance-id form, a GPU-UUID prefix, an ssh endpoint, a public IP (after
    `public_ipaddr` or `@`, or in an environment dump), or a home directory.
