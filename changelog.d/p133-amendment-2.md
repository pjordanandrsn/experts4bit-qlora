### P133 Amendment 2: the reading's guard is 2.0 h, and its host needs 1 Gbps down

- `p133-4090-1` was aborted before any arm ($0.50; nothing read). A 17-min fetch and a slow bake left the registered
  time-left check no room for the quality phase, so the lane would have exited with no verdict.
- The reading's guard goes from 1.25 h to 2.0 h, and the pin rule adds Vast `inet_down >= 1000` Mbps. The time-left checks
  and every staged file are unchanged. The rerun `p133-4090-2` costs at most $1.65, and the lane at most $2.32 of its $3.75
  ceiling.
