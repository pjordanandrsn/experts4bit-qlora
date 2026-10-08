### P117 read (#846): decoding 32 or 64 rows in one graph costs no measurable quality; AT_PARITY (bench, docs and register only)

- **The reading** (`p117-5090-1`, one RTX 5090, $0.914): on SC2e's int4 stack (Qwen3-30B-A3B), teacher-forced over 64
  wikitext windows decoded together, one 64-row piece reads −0.0032 nats against four 16-row pieces
  (`e4b.serve.p117.wide-bucket-quality.qwen3.5090.2026-10-08`). 32-row pieces and a padded 64-row step pass P110's bar
  too; the captured 64-row replay emits the eager tokens at every position; a halved decode scale fails the bar.
- **What it licenses:** `E4B_PAGED_BUCKETS=auto` as `serve_paged`'s default, with SC2e's capacity read, in a separate PR.
- **Files:** `bench/p117/RESULTS-p117.md`, `bench/p117/receipts/p117-5090-1/` (with `SHA256SUMS`), `docs/STATUS.md`,
  the register row.
