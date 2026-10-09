### P126 read (#846): DEFAULT_ON_4 — the tile table over 4 programs makes Qwen3-30B-A3B's 64-row decode step about 15.5 % faster

Attempt 2 (`p126-5090-2`, under Amendment 1, on another host) read **DEFAULT_ON_4**:
- **P = 4 is licensed.** Its blocks read 0.8442 and 0.8456, and every token was identical to P = 1's.
- **P = 8 was ineligible.** Its blocks disagreed by 3.64 %, through the same one-runner level offset as attempt 1.
- **The premise.** The one-program table was 18.0 % of the eager step.

The recurring offset is recorded as an open measurement question, with a proposed check for future interleaved lanes.
The default flip is its own PR. Register row `e4b.serve.p126.tile-programs.qwen3-int4.5090.2026-10-09`;
`bench/p126/RESULTS-p126.md`.
