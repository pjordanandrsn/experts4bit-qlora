### P121 read (#846): K25 is LICENSED on Qwen3-30B-A3B's served W16 step — 1.57× as fast as the NF4 M-tile, within P110's quality bar (bench and docs only)

- **The reading** (`p121-5090-1`, one RTX 5090, $1.056; lane $1.245): on the default `serve_paged` server, K25
  (`E4B_NF4_GROUPED_SMALLM=auto`, the default since P96) decodes 16 requests at 771 tok/s against 491 for the M-tile
  (`0`): 20.7 against 32.6 ms a step, g16 = 1.5697. W1 is unchanged (0.998) with identical tokens.
- **Quality** at 128 routed rows a step, P115 Phase B's teacher-forced instrument: bias +0.0002 nats on wikitext and
  +0.0022 on c4val1, against bars of 0.0115 and 0.0116. The mutant moved +1.06 and +0.78. Q1–Q5 held.
- **No code change:** the default was already `auto`. Register row `e4b.serve.p121.k25-w16.qwen3.5090.2026-10-08`.
  STATUS's K25 row and FAM0's `qwen3_moe` K25 cell now cite it.
- **Files:** `bench/p121/RESULTS-p121.md`, `bench/p121/receipts/p121-5090-1/` (with `SHA256SUMS`).
