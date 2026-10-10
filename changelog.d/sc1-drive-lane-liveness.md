### SC1's controller detects dead lanes by PID identity and records host reboots (bench and tests only)

- `bench/sc1/sc1_drive.sh` adopts `bench/common/lane_liveness.sh`, as the tc1, p127, fam and census drivers did
  in #1512 and #1517:
  - the launch returns the box script's PID and its `/proc` identity;
  - each poll probes that identity rather than counting `bash sc1_run.sh` processes;
  - a host reboot (boot id changed, or uptime below the lane's age) ends the wait and writes `host-fault.json`, so
    the launcher's receipt carries the fault.
- `tests/test_lane_liveness.py` now runs sc1_drive.sh's own poll loop and launch handshake with the other drivers.
  `tests/test_sc2g_box.py`'s A1 check asserts that no process-name count remains.
