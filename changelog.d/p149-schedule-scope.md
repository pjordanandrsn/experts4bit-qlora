### P149: the `max_new_tokens` dependence placed in the decode schedule

`bench/p149/RESULTS-p149.md` gains "The schedule dependence (follow-ups, 2026-10-21)". Two follow-ups after the run
(`p149mnt-5090-6` and `-7`, a one-layer cut on one RTX 5090) place the QUALIFIER_VOID cause in the decode schedule. The
fp8 paged attention's split-KV count changes with the graph bucket a step replays. With one split at every bucket,
buckets 2 to 16 match bit for bit, and a small bucket-1 difference remains, not attributed. A routing-matched read must
match the decode schedule. The `p149-5090-1` run-id row now says the ask was refused at create while still listed. No
verdict, gate or claim moves.
