### RD1 amendment 2 registered: RTX 5090 on RunPod Secure, with a proving run first; the runner installs rsync (bench and tests only)

- **Why.** Under amendment 1's load-gated anchor, a fourth Vast draw (`rd1-5090-5`) landed on machine 145701 a third time.
  - Host load1 stayed between 6.5 and 47 for 35 minutes on a 256-thread host, so the 5.0 gate was never reached.
  - The anchor failed launch and H2D stability even at load 12.
  - Four Vast draws cost $0.43 and produced no reading.
- **What.**
  - The lane moves to RunPod Secure Cloud's RTX 5090 (the provider rate row is adertha-agents#173).
  - `rd1_run.sh` installs `rsync`: `tc1_drive.sh` fetches with it, and RunPod's pytorch image has none (`tc1c-h100-19`
    fetched zero files).
  - A proving run (`RD1_PROVE=1`, forwarded by `tc1_drive.sh`) runs install, tripwire, the load-gated anchor and host-load
    sampling, with no probe. It reads the fetch path, the anchor and whether the 5.0 gate is reachable before any draw.
  - `RD1_REHEARSAL` is still never forwarded.
- **Tests.** `tests/test_rd1_lane.py`: exactly `RD1_PROVE` of RD1's knobs is forwarded, and rsync is installed before the
  anchor.
