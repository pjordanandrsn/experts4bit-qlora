### Host-facts probe: the drive exits through the shared tail (Amendment 2)

- `bench/hostprobe/hostprobe_drive.sh` now records the box's exit code and a success marker beside its log, and exits
  through `bench/common/drive_tail.sh`. It leaves the drive-tail allowlist.
- Exits: 0 and 22 are as registered. The host codes 13, 14, 17 and 18 pass through. A facts line with a non-zero exit
  is now 26, not 0, though the facts are still saved. No probe has run, so no receipt reads differently.
