### Added

- `tests/test_public_rederivation.py` and a `Re-derive:` line convention (CONTRIBUTING.md). A receipt that a reader can
  rebuild names one command and the committed file it writes; the test runs every such line on CPU from the committed
  files and requires byte-identity, so a later reducer edit that stops an old verdict reproducing fails CI. It is seeded
  with 11 lines, each checked before it landed: hbd-5090-1, hpn-a2000-read8, p149-5090-2, p158-5090-1, p160-5090-2 and
  p161-5090-1 (whose READMEs already gave a command), and p143-5090-2, p143a2-5090-1, p143a3-5090-1, p150-5090-1 and
  p156-5090-1 (whose verdicts were re-derived only from the private receipt store until now). A one-byte edit to a
  committed verdict fails the test. No receipt file is changed: a committed receipt is never edited, so each line lives
  in a new lane-level `bench/<lane>/REDERIVE.md` (hbd, hpn, p143, p149, p150, p156, p158, p160, p161), and a further test
  flags any `Re-derive:` line that a file under `receipts/` did not carry at its first commit.
