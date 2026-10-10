### Q235: the pre-rental fetch gate (gates 3 and 5), enforced by the drive

- `bench/q235/q235_fetch_gate.py` resolves from a CPU host everything the box will fetch:
  - the hub listing at `ac9c66cc` must equal `model_files.json` row for row, and each file's resolve HEAD must answer the
    commit, size and etag (lane P130's resolvers and listing checks);
  - every locked wheel URL must answer a one-byte GET made with the box's own opener (`q235_bootstrap.fetch_wheel`);
  - grouped-nf4-gemm and this repository must fetch at their pinned commits.

  It also scans the sealed closure, the locks, the drive and the registration for unfilled values.
- `q235_drive.sh` refuses to launch without a passing report for its own commit under a day old, and copies the report into the
  run. `wheels.json` names what pins this repository (`E4B_SHA`) instead of carrying a stand-in value.
