### Driver tails: the box's exit code reaches the launcher, and success needs the box's marker (a CPU test over every driver)

- `tests/test_drive_tail_codes.py` runs every `bench/**/*_drive.sh` fetch-and-exit tail on synthetic fetched directories. It
  asserts three things:
  - the host codes 13, 14, 17 and 18 pass through unchanged, since the launcher excludes a machine only on the driver's own exit
    code;
  - rc 0 without the success marker never exits 0;
  - a missing exit file never exits 0.
  A control (rc 0 with the marker exits 0) proves the harness drives the tail. The tail runs with no `rm`, `ssh`, `scp`, `rsync`
  or `curl` on PATH.
- 172 of today's 173 drivers fail at least one assertion:
  - 148 exit 0 on rc 0 without the marker;
  - 4 (P227, P228, P229, P236) map every box code to 23;
  - 20 have no fetched exit file the harness can drive.
  They are listed in `tests/fixtures/drive_tail_allowlist.json`, each with its sha256, the assertions it fails, and its status.
  131 are closed lanes (results merged); 41 are marked live, to be fixed in the lane's next PR.
- A new or changed driver is never allowlisted, and an entry whose driver starts to pass must leave the list.
