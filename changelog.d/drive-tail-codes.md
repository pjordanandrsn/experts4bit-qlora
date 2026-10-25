### Driver tails: the box's exit code reaches the launcher, and success needs the box's marker (a CPU test over every driver, and a shared tail)

- `tests/test_drive_tail_codes.py` runs every `bench/**/*_drive.sh` fetch-and-exit tail on synthetic fetched directories. It
  asserts three things:
  - (a) the host codes 13, 14, 17 and 18 exit unchanged, since the launcher excludes a machine only on the driver's own exit code;
  - (b) rc 0 without the success marker never exits 0;
  - (c) a missing exit file never exits 0.
  A control (rc 0 with the marker exits 0) proves the harness drives the tail. Tails run with only read-only tools on PATH.
- `bench/common/drive_tail.sh`'s `drive_finish OUT NONCE PREFIX "<the lane's registered codes>"` is the shared tail:
  - 13/14/17/18 always pass through, plus the lane's registered codes;
  - 23 for a missing or unreadable exit file;
  - 24 for rc 0 without this nonce's success marker;
  - 26 for any other code.
  A live lane fixes its tail by sourcing it.
- Today's failing drivers are listed in `tests/fixtures/drive_tail_allowlist.json`, each with its sha256, the assertions it fails,
  and its status (closed lane, or live: fixed in the lane's next PR). A new or changed driver is never allowlisted, and an entry
  whose driver starts to pass must leave.
- Wrappers that read no fetched exit file and delegate to another driver are listed by path. `bench/sc2/sc2_qualify_drive.sh`
  delegates to `bench/sc1/sc1_drive.sh` and checks its marker (e4b#2139). No other script may run `sc1_drive.sh`.
