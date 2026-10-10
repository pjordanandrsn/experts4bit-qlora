### SD2 Amendment 5b (#1313, operational): canary for E/Q, fail-soft reporting offline, V re-joined

- **Why.** `sd2-5090-2` ($0.713) crashed in V0 on a reporting-only statistic. Under the two-strikes rule the read path
  is simplified, with a net deletion and no new guard.
- **Depth reporting leaves V0.** V0 keeps only its strict values; the raw key-row tensors go to `v_angles.pt`; the
  reducer computes the depth report offline and fail-soft. V0 and V1 stay in one process.
- **The canary** runs only the long paths: one E arm, and one Q window when Q will run. A failure stops the lane at
  rc 26 before the paid stages.
- **Unchanged:** the rule and every verdict input. The guard is 3.75 h, with a ceiling of $3.87.
