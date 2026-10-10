### TC harness: T21 reads an offloaded expert stack; a failed drive names its reason

- **T21 under expert offload.** `frozen_base_probe` dequantised expert 0 from the base module, which holds 0-element
  placeholders between forwards when the experts are offloaded, so offloaded rows recorded
  `experts: IndexError: index 0 is out of bounds for dimension 0 with size 0` and probed `q_proj` alone. The probe now finds the
  offload handle on the wrapper chain and reads its home copy of the packed storage and absmax, staged to the handle's device and
  dequantised there as the forward does, with the byte-flip control on that copy (`source` says so; `experts_offloaded` records
  the residency). A 0-element stack with no handle to resolve it is an error row for the slot, not a probe of nothing. Resident
  rows are unchanged. C1, the frozen hasher, already read the offload homes (TC3 amendment 2), so no registered reading moves.
- **The drive's error line.** On any non-zero exit `tc1_drive.sh` writes `command-error.txt` in the run directory: the exit code,
  its own last message and, once the box's artifacts are fetched, the box's last error line (`lane_box_error_line`,
  `lane_write_command_error` in `bench/common/lane_liveness.sh`). The rental launcher puts that line in the receipt's notes after
  `command exited N`. Reporting only: the exit code is unchanged.
- **Tests.** `tests/test_tc1_arm.py`: the offloaded probe reads the same bytes as the resident one, and an unresolvable
  placeholder is an error. `tests/test_lane_liveness.py`: the line's content and cleaning, the box-log choice, and the drive's
  refusal written with its exit code intact.
