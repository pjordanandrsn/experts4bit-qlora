### RD1 amendment 3 registered: a post-probe anchor licenses the reading, and host load is informational (bench and tests only)

- **Why.** RD1's RunPod proving run (`rd1-rp-prove-1`, $0.23) fetched every file, and the anchor passed on its first
  attempt at host load1 19.58. But load1 never reached amendment 1's 5.0 gate (min 9.66, on 120 CPUs).
  - Across five anchored boxes, load did not predict the anchor: Vast's 145701 failed at 12–31, RunPod passed at 19.6.
- **What.**
  - `rd1_run.sh` drops the load wait.
  - It runs the anchor before the probe (at most 3 attempts) and once more after it, and writes the post-probe verdict
    into the probe's receipt as `anchor_post`.
  - `rd_table.py` decides only when that post-probe anchor passed. Host load stays recorded, informational only.
- **Tests.** `tests/test_rd1_lane.py`: a passing post-probe anchor decides even at load 30, and a failed or missing one never
  does.
