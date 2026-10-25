# Receipt redaction: rented-hardware identifiers

A public receipt never needs the rented card's GPU UUID or the provider's instance id. Both reached the public tree
before the receipts rule said so (2026-10-20). On 2026-10-25 they were replaced, in place and forward only, by stable
keyed tokens. Git history was not rewritten, and the private receipt store keeps every original byte-exact.

## What changed

Each distinct value became one token: the first 8 hex of HMAC-SHA256 under a key that only the private receipt store
holds.

| was | now |
|---|---|
| `GPU-<8-4-4-4-12 hex>` (nvidia-smi's uuid) | `GPU-redacted-<8 hex>` |
| `"uuid": "<8-4-4-4-12 hex>"` (torch's form) | `"uuid": "redacted-<8 hex>"` |
| `"vast_instance_id": "<6-9 digits>"` | `"vast_instance_id": "instance-redacted-<8 hex>"` |
| `Vast instance <7-9 digits>` | `Vast instance redacted-<8 hex>` |

- **Tokens, not blanks.** Equal values stay equal and different ones stay different. `bench/ra/ra_reduce.py` puts the GPU
  UUID in its run identity, and a null would make every card look alike. No two values share a token: the script
  refuses to run if one would.
- **The shape is not a UUID's.** That lets `.github/private-markers.txt` refuse the UUID shape across the whole tree,
  not only in added lines. An instance id is refused in added lines, since tests keep synthetic ones on purpose.
- **Scope:** 2155 files, 137 GPU UUIDs and 295 instance ids. Machine ids are not touched here.
- **The transform** is `scripts/redact_receipt_ids.py`, deterministic given the key. `llms-full.txt` is not edited by it:
  it is rebuilt from the redacted files (`python scripts/build_llms_bundle.py`).

## The claims register

`docs/claims.json` named instance ids in 353 claims' `hardware` text (and once each in a `notes` and a `claim` text).
They were redacted like any other file. The register's rule is that a claim is never edited, only superseded. That
rule protects a claim's substance: its number, meaning, unit, status, tier and evidence. A host descriptor's instance id
is none of those, and superseding 353 rows to change it would bury the register. So the edit is proved, not asserted:
`tests/test_receipts_redaction.py` loads the register at v0.58.1 and now, and checks two things. Every claim's id,
order and every other field are equal. Its `hardware`, `notes` and `claim` text differ only by
`instance <digits>` becoming `instance redacted-<8 hex>`, with one token per id. `evidence_private` is not touched.

## A second edit, on three lines

Three redacted lines also named the control plane or the receipt store the way public text no longer does. Redacting
a line makes it an added line, which the guard reads, so the same transform rewords each name to the project-voice
form (the script's `REWORD` table; each pair must match exactly once):

- `bench/h2h-2026-10-02/tc1/README.md`, the `tc1-5090-44` row: a control-plane issue reference becomes "the control
  plane's issue #138".
- `bench/h2h-2026-10-02/tc1/RESULTS-tc1-steady18.md`, the `tc1-5090-48` row: likewise, "the control plane's issue #139".
- `bench/p100/RESULTS-p100.md`, `p100-5090-2`'s receipt line: the receipt store's repository name becomes "the receipt
  store".

`--verify` re-derives these three with the rewording, so their manifest rows hold the final bytes.

## How to check it

- **Each file's two hashes.** [`receipts-redaction.sha256`](receipts-redaction.sha256) lists every edited file: its
  original sha256, its redacted sha256, and where the original is held.
  - `store`: the private receipt store holds the original byte-exact (1900 files).
  - `git:605d21e6`: the original is only in this repository's history at v0.58.1 (255 files). These are mostly
    documents and code written in the repository.
- **Originals are git's bytes at v0.58.1.** In both cases the original is the file at `605d21e6`.
  `python scripts/redact_receipt_ids.py --key-file K --verify docs/receipts-redaction.sha256 --from-rev 605d21e6`
  re-derives every redacted file from it and checks both hashes. Anyone holding the key can run it.
- **Receipts with a `SHA256SUMS`:**
  - each edited file's line is re-pinned to its redacted bytes;
  - its store-held original is named in that receipt's `STORE-ONLY.sha256` under a `# same bytes: <path> at
    605d21e6` line, which `tests/test_receipt_sums.py` checks against git.
- **Older receipts with no `SHA256SUMS`** are covered by the manifest alone.

## What was left as it was (44 files)

A file whose bytes another file pins by hash, or that a runner reads, keeps its identifiers. Editing it would silently
break its seal. `tests/test_receipts_redaction.py` pins each one's sha256. The guard exempts the ones that still carry
the UUID shape only while they keep that pin (`#exempt-fixture:` in `.github/private-markers.txt`), and the test holds
the guard's list to exactly those:

- **Sealed or hash-pinned (35).** The two sealed DQ11 amendments (`science.sha256`); dq7-5090-1's fourteen
  `read-*.json`, whose sha256s its own `d7-unlicensed.json` records; the fifteen train-parity tp1 `*_train_*.json`, whose
  sha256s P67's `reread.json` records; `bench/p132/p132_run.sh` (`staged-p132.sha256`); and three test-pinned files:
  sd2-5090-3's `forensics.txt`, `tests/fixtures/p132_canary/forensics.txt` and
  `tests/fixtures/dq11-census-stalled/receipt.json`.
- **A runner reads the value (2 more).** P132's same-card refusal compares the live card's UUID to a registered
  reference, so `bench/p132/PREREG-p132.md` and `tests/test_p132_card.py` keep the real value with the rest of the P132
  set.
- **Test sources (7).** Six tests feed synthetic identifiers to the code that strips them (the HO2, HO3 and TC1 public
  projections, the GPU contract, the ssh known-hosts driver, P41's drivers). `tests/test_dq11_l_census_launch.py` names
  the instance its pinned fixture names. Their values are inputs, not records.

## Registrations edited

These PREREGs named a smoke run's instance or card. The identifiers were descriptive, not part of what was registered,
so the registered science is unchanged:

- `bench/fam/PREREG-fam.md`
- `bench/p133/PREREG-p133.md`
- `bench/p43/P43-PREREG.md`
- `bench/p44/P44-PREREG.md`
- `bench/p48/P48-PREREG.md`
- `bench/p55x/P55X-PREREG.md`
- `bench/sd2/PREREG-sd2.md`
- `bench/sv7/SV7-PREREG.md`
