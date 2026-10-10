### Rental drivers record a detected host reboot as the launcher's host-fault marker (bench and tests only)

- When #1512's liveness check reads a reboot, the TC1, P127, FAM and locality drivers write `host-fault.json` into the
  launcher's run directory: `kind` reboot, both boot IDs and the uptimes as evidence.
- The launcher (adertha-agents#204) copies it into the receipt, so a run that dies on a host reboot can exclude that
  machine from the next launch. p127-prove-3 re-bought the host that rebooted under p127-prove-2.
- `bench/common/lane_liveness.sh` gains `lane_reboot_evidence` and `lane_write_host_fault`. The marker is written
  atomically, and its JSON survives quotes, backslashes and control characters.
