### serve_paged: `E4B_PAGED_BUCKETS=auto` by default, so the widest decode step is one graph replay (lanes SC2e and P117; `1,2,4,8,16` restores the old list)

- **The licence.** SC2e (#846) read 64 slots holding the SLO to 12 req/s with `auto` against 8 on the old list, whose
  64-row step runs as four 16-row replays. P117 read the quality AT_PARITY: one 64-row piece is −0.0032 nats against
  four 16-row pieces (`e4b.serve.p117.wide-bucket-quality.qwen3.5090.2026-10-08`), inside the 16-row arithmetic's own
  neutral perturbations.
- **The change.** `E4B_PAGED_BUCKETS` unset or empty reads `auto`: every power of two below `max_seqs`, then `max_seqs`.
  Up to 16 slots that is the old list exactly. With `E4B_PAGED_MAX_SEQS=auto` the widths are now 64, 32 or 16; on an RTX
  5090, Qwen3-30B-A3B int4 at the server's default 4,096 tokens a slot resolves to 32 slots (16 before).
- **The way back.** `E4B_PAGED_BUCKETS=1,2,4,8,16`.
- **Not changed.** A `PagedServeConfig` built in code keeps the list unless `buckets="auto"` is passed; `ServeSetup`
  plans (and loggetta's) still write the list explicitly.
- **Scope.** Measured on Qwen3-30B-A3B int4 on an RTX 5090; other models get the wide buckets on the strength of this
  read, not their own. Outputs under load change more often with wide steps, at no measured quality cost.
