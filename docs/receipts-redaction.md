# Receipt redaction: rented-hardware identifiers

A public receipt never needs the rented card's GPU UUID, the provider's instance id, or the box's ssh endpoint or public
address. All four reached the public tree before the receipts rule said so (2026-10-20). On 2026-10-25 they were
replaced by stable keyed tokens, in place and forward only. Git history was not rewritten, and the private receipt store
keeps every original byte-exact.

This is the second cut. The first one (2026-10-25, superseded before it merged) checked its own completeness with the
transform's own matcher, so it could only confirm what the transform already knew, and about 170 values stayed raw. This
cut is driven by values, not shapes, and is checked three independent ways (see [How to check it](#how-to-check-it)).

## What changed

Each distinct value became one token: the first 8 hex of HMAC-SHA256 under a key that only the private receipt store
holds. The key that cut the first attempt is retired, since that attempt's head paired some tokens with raw values.

| was | now |
|---|---|
| `GPU-<8-4-4-4-12 hex>` (nvidia-smi's uuid), or its first 8 hex | `GPU-redacted-<8 hex>` |
| `"uuid": "<8-4-4-4-12 hex>"` (torch's form) | `"uuid": "redacted-<8 hex>"` |
| `"vast_instance_id": <digits>` | `"vast_instance_id": "instance-redacted-<8 hex>"` |
| `Vast instance <digits>`, `box <digits>`, `vast <digits>` | `Vast instance redacted-<8 hex>`, `box redacted-<8 hex>`, ... |
| `instance<digits>`, `box<digits>` (glued) | `instance-redacted-<8 hex>`, `box-redacted-<8 hex>` |
| `sshN.vast.ai:<port>` | `ssh-redacted-<8 hex>` |
| a rented box's public IPv4 | `ip-redacted-<8 hex>` |
| a home directory, an agent's scratch directory | `<home>`, `<scratch>` |

- **Values, not shapes.** The transform (`scripts/redact_receipt_ids.py`) takes the private store's token map: every
  value the store records for every box it rented, plus the few that only the public tree ever recorded. It replaces
  each value where it is unambiguous: a GPU UUID anywhere, an instance id only after an id word (instance, vast, box,
  contract, container, provider, list_after) within 48 characters. Seven files name ids without the word, in tables
  and lists, so the word is not required there (`ID_FILES`: CHANGELOG.md, docs/claims.json, the P40 and P55
  registrations, and three RESULTS files). A number inside a longer number or a float is never touched.
- **Tokens, not blanks.** Equal values stay equal and different ones stay different. `bench/ra/ra_reduce.py` puts the
  GPU UUID in its run identity, and a null would make every card look alike. No two values share a token: the script
  refuses to run if one would. A UUID's 8-hex prefix takes the whole UUID's token.
- **Scope:** 2220 files edited, 7 binaries removed, 1 archive re-packed. Together they held 138 GPU UUIDs, 341 instance
  ids, 3 ssh endpoints and 2 box addresses, plus home and scratch directory paths. Machine ids and offer ids are not
  touched here: they name the provider's machine or its marketplace listing, not the rental, and have their own
  follow-up.
- **The shape is gone.** That lets `.github/private-markers.txt` refuse the UUID shape across the whole tree, and
  `scripts/check_public_ids.py` refuse every identifier form above in every added line, binaries included.
- `llms-full.txt` is not edited by the transform: it is rebuilt from the redacted files
  (`python scripts/build_llms_bundle.py`).

## Binaries removed, and one archive re-packed

A binary can carry an identifier split across chunks or compressed past any scan, so binaries are never edited. The
seven that carried identifiers left the tree. Each is in the private receipt store byte-exact, and a `STORE-ONLY.sha256`
beside where it was names its sha256:

- `bench/cpu-router/g1_audit_legacy.nsys-rep`, `g1_audit_replica.nsys-rep`
- `bench/hybrid-g3/g3_nsys.nsys-rep`, `g3_nsys_235b.nsys-rep`
- `bench/hybrid-g6/g6_nsys.nsys-rep`
- `ab-telemetry/wandb/wandb/offline-run-*/run-*.wandb` (two)

The text extracts committed with the traces (`*_nsys_api*.txt`, `*_streams.txt`) stay. Each RESULTS file that names a
trace now says it is store-only.

`bench/dq9/receipts/dq9-5090-2/raw-science.tar.gz` is a reproduction input, so it was re-packed rather than removed.
Its 33 receipts with identifiers were redacted like any file and `SCIENCE-SHA256SUMS` re-cut for those members only.
The archive is deterministic: sorted members, zero mtimes and owners, gzip level 9 (`scripts/repack_tar.py`). The
receipt's own reproduction recipe still passes on it: every `SCIENCE-SHA256SUMS` line verifies, and `dq9_reduce`'s
output is byte-identical to the original archive's. The original archive is in the private receipt store.

## The claims register

`docs/claims.json` named instance ids in 390 claims: 383 `hardware` texts, 57 `notes` and 35 `claim` texts. They were
redacted like any other file. The register's rule is that a claim is never edited, only superseded. That rule protects
a claim's substance: its number, meaning, unit, status, tier and evidence. A host descriptor's instance id is none of
those, and superseding 390 rows to change it would bury the register. So the edit is proved, not asserted.
`tests/test_receipts_redaction.py` loads the register at the originals' commit and now, and checks two things:

- every claim's id, its order, and every field outside those three texts are equal;
- each changed text, with every token turned back into a slot for the identifier shape it replaced, matches the
  original exactly, and one value always took one token.

`evidence_private` is not touched.

## Reworded lines

Eight lines were also reworded by the transform (its `REWORD` table; each pair must match exactly once). `--verify`
re-derives them, so their manifest rows hold the final bytes.

- **Three named the control plane or the receipt store** the way public text no longer does. Redacting a line makes it
  an added line, which the guard reads, so each name becomes its project-voice form:
  - `bench/h2h-2026-10-02/tc1/README.md`, the `tc1-5090-44` row: "the control plane's issue #138".
  - `bench/h2h-2026-10-02/tc1/RESULTS-tc1-steady18.md`, the `tc1-5090-48` row: "the control plane's issue #139".
  - `bench/p100/RESULTS-p100.md`, `p100-5090-2`'s receipt line: "the receipt store".
- **Four RESULTS files named a removed trace:** `bench/cpu-router/RESULTS-hybrid-phase1.md`,
  `bench/hybrid-g3/RESULTS-g3-shakeout.md`, `bench/hybrid-g3/RESULTS-g3-formal.md`, `bench/hybrid-g6/RESULTS-g6.md`.
  Each now says the trace is store-only.
- **dq9-5090-2's README** described its archive as holding "unchanged JSON receipts". It now says the identifiers were
  replaced.

## How to check it

**Three independent checks** say nothing was missed:

1. **An oracle from the store's records.** Every value in the private token map is searched for, case-insensitively,
   in every tracked file's raw bytes, inside every archive one level down, and in file names. It found none outside the
   holds below. 81 numbers merely equal an id: 79 sit inside a longer number (a digit on at least one side, none
   bounded only by a `.`), and 2 are `/proc/stat` cpu jiffy columns, confirmed by their position. A second reviewer
   checked each one against the raw bytes, and they were left.
2. **A shape-and-context check in CI** (`tests/test_receipts_redaction.py::test_no_identifier_outside_the_holds`). It
   never consults the map, so it catches what the map cannot know. It finds GPU-/MIG- UUIDs and any 6-hex prefix of one;
   UUIDs after a `uuid` key; container labels; provider ssh endpoints; a public IPv4 after an `@`, an address word or a
   box token (loopback, private and documentation ranges pass, and so do dotted package versions); and any 7-10 digit
   number within 24 bytes after an instance word. It reads raw bytes and gzip members. Its exceptions are named in
   `ALLOWED` with a count and a reason: one offer id inside the provider's own error text (for the offer-id follow-up),
   two coincidences in tests, and a test's synthetic addresses, which it feeds to the code that strips them from logs.
   It is what found the 13 instance ids and the ssh endpoint that only the public tree had recorded.
3. **The byte guard on every change** (`scripts/check_public_ids.py`, run by the private-marker guard). `git grep -I`
   skips binaries, so this reads bytes: the whole tree for UUIDs, and every added line for each form above.

**Each file's two hashes.** [`receipts-redaction.sha256`](receipts-redaction.sha256) lists every edited or removed file:
its original sha256, its redacted sha256 (`removed` for the seven binaries), and where the original is held.

- `store`: the private receipt store holds the original byte-exact (1903 files, and the seven binaries).
- `git:069a029b`: the original is only in this repository's history (317 files). These are mostly documents and code
  written in the repository.

**Originals are git's bytes at `069a029b`.** Anyone holding the token map can re-derive every redacted file from them
and check both hashes:

```
python scripts/redact_receipt_ids.py --map M --verify --rev 069a029b --holds H
```

**Receipts with a `SHA256SUMS`** (136 of them):

- each edited file's line is re-pinned to its redacted bytes;
- its store-held original is named in that receipt's `STORE-ONLY.sha256` under a `# same bytes: <path> at 069a029b`
  line, which `tests/test_receipt_sums.py` checks against git.

**Older receipts with no `SHA256SUMS`** are covered by the manifest alone.

## What was left as it was (44 files)

A file whose bytes another file pins by hash, or that a runner reads, keeps its identifiers. Editing it would silently
break its seal. `tests/test_receipts_redaction.py` pins each one's sha256. The guard exempts the ones that still carry
the UUID shape only while they keep that pin (`#exempt-fixture:` in `.github/private-markers.txt`), and the test holds
the guard's list to exactly those.

- **Sealed or hash-pinned (35):**
  - the two sealed DQ11 amendments (`science.sha256`);
  - dq7-5090-1's fourteen `read-*.json`, whose sha256s its own `d7-unlicensed.json` records;
  - the fifteen train-parity tp1 `*_train_*.json`, whose sha256s P67's `reread.json` records;
  - `bench/p132/p132_run.sh` (`staged-p132.sha256`);
  - three test-pinned files: sd2-5090-3's `forensics.txt`, `tests/fixtures/p132_canary/forensics.txt` and
    `tests/fixtures/dq11-census-stalled/receipt.json`.
- **A runner reads the value (2 more).** P132's same-card refusal compares the live card's UUID to a registered
  reference, so `bench/p132/PREREG-p132.md` and `tests/test_p132_card.py` keep the real value with the rest of the P132
  set.
- **Test sources (7).** Six tests feed synthetic identifiers to the code that strips them: the HO2, HO3 and TC1 public
  projections, the GPU contract, the ssh known-hosts driver, and P41's drivers. `tests/test_dq11_l_census_launch.py`
  names the instance its pinned fixture names. Their values are inputs, not records.

## Registrations edited

These registrations named a run's instance, card or box. The identifiers were descriptive, not part of what was
registered, so the registered science is unchanged:

- `bench/dq10/AMENDMENT1-HOST.md`
- `bench/fam/PREREG-fam.md`
- `bench/h2h-20260905/p37/PREREG.md`
- `bench/h2h-20260906/tp2/P40-PREREG.md`
- `bench/p112/PREREG-p112.md`
- `bench/p133/PREREG-p133.md`
- `bench/p43/P43-PREREG.md`
- `bench/p44/P44-PREREG.md`
- `bench/p48/P48-PREREG.md`
- `bench/p55/P55-PREREG.md`
- `bench/p55x/P55X-PREREG.md`
- `bench/p56/P56-PREREG.md`
- `bench/sd2/PREREG-sd2.md`
- `bench/sv7/SV7-PREREG.md`
- `bench/vsplit/PREREG-vsplit.md`
