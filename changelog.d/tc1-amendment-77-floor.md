### TC1 amendment 77: amendment 76's host floor screens single-thread speed, under a 3 h guard

- `qwen3pos76co`'s pre-flight no longer screens the container's CPU allotment. It times the launcher's own `cpu_loop_mops` loop (the same
  body as P225's probe), the median of 5 runs of at least 2 s, and refuses under 15 Mops (rc 18) before any venv or fetch.
- The floor is sized from box tc1-5090-155's 25.91 Mops with amendment 76's phase table, every arm run three times and 25 % headroom,
  for a 3 h guard. The worst case moves from $3.20 to $4.25.
- `bench/tc1/tc1_run.sh` (`single_thread_mops`, `POS77_MIN_MOPS`); amendment 76's floor and budget lines are marked superseded.
  `tests/test_tc1_pos76_family.py` covers the floor (refused under 15 or unreadable, passed at 15 and above, other families untouched)
  and checks that the probe is the launcher's loop.
