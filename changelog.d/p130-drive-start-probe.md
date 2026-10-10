### P130's driver probes a dropped start instead of failing blind (bench and tests only)

- **Why.** p130-prove-1 (2026-10-10) cost $0.001 and measured nothing. The box's ssh proxy closed the lane-start
  connection after its banner, so the driver could not tell whether the child had started, and failed (21).
- **Now.** `bench/p130/p130_drive.sh` never resends the start. Up to three fresh, read-only connections probe for THIS
  run's nonce:
  - a child that bound it is adopted, and the run continues;
  - one that bound it and already exited leaves its files to fetch;
  - anything else is the old failure (21).

  A child that exited early (125), a handshake timeout (124) and a failed `cd` (20) fail as before, without a probe.
- **P127's driver** shares that start step. It gets a note, not the change: it runs no further rental.
- `tests/test_p130_drive_start.py` runs the real driver against fake `ssh`, `scp` and `rsync` in five cases. The start
  is sent exactly once in every case. Run against the previous driver, four of the five fail.
