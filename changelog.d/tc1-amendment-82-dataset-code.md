### TC1 amendment 82: a dataset refusal is lane code 29, never the launcher's disk code 13

- `tc1_run.sh` used to finish 13 when the alpaca fetch failed or a dataset's sha256 missed its pin. The launcher reads 13 as disk and
  excludes the machine, so an upstream mismatch would have excluded every host in turn. Those three sites now finish 29. The disk floor
  stays 13, and `tc1_drive.sh` registers 29 for `drive_finish`. Not 21: the driver itself exits 21 on a start failure.
- `tests/test_tc1_exit_codes.py` reads the scripts' own text and holds four rules, each with a mutant:
  - every `finish N` is a host code with a host-floor reason, or a TC1 code;
  - 13 appears only at the disk floor;
  - the driver's lane list is exactly the script's codes;
  - no lane code is one the driver exits on its own, except 0 and 78.
