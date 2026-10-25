### Driver tails: the GPU-contract driver hands the launcher the box's code; 27 allowlisted lanes read as closed (tests and one driver's tail only)

- `bench/gpu-contract/gc_drive.sh` ends with the shared `drive_finish` (`bench/common/drive_tail.sh`). The launcher's host codes 13,
  14, 17 and 18 pass through, and so do the codes `gc_run.sh` exits with (9, 10, 15, 28, 30, 78, 130). A box rc 0 without this
  nonce's `GC_SUCCESS` marker now exits 24 instead of 0. A missing or unreadable exit file exits 23, and any other code exits 26.
  The driver refuses with 78 when the shared tail is missing.
- The same driver now follows bench/common rule 8: its rsync fetch carries `--no-links`, and a link in the fetched artifacts
  (the dry run's tar stream could carry one) refuses with 22. Its entries in `tests/fixtures/drive_tail_allowlist.json` and
  `tests/fixtures/rule8_legacy_fetch.json` are removed.
- `tests/test_gpu_contract.py` runs the whole driver as a `GC_DRY=1` dry run on a stub box, with a stand-in runner. That covers
  staging, the nonce handshake, the poll, the fetch and the tail. It asserts rc 0 with the marker gives 0, rc 0 without it gives
  24, each host code passes through, the contract's 30 passes through, an unregistered code gives 26, and a fetched link gives 22.
- 27 rows of the driver-tail allowlist move from live to closed. Each row names its record: a grouped-nf4-gemm `kernel/RESULTS-*`
  file for the 14 kernel lanes (B374, B393, K14 through K34), a closed work item, a merged read, a superseding registration, or a
  precondition that cannot hold. Only status strings change, and every sha256 and fails list is unchanged.
