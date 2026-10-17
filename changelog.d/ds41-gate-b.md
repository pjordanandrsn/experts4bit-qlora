### Added

- `bench/ds41-gate-b/`: gate B for DeepSeek-V4.1-Flash-1017, an analytic memory, residency and transfer model ($0; no weights,
  no build). The Engram tables (203 GB, 12.7 KB per token) belong on disk unless RAM is plentiful, leaving the RAM to the
  289 GB of experts: full host residency then needs about 298 GB. Decode transfer bounds at batch 1 and 128K: 0.44 tok/s on the
  A2000 host, 4.87–7.11 tok/s on Q235's 239 GB-floor boxes and up (H2D-bound at gen 4). The tables' random-read rate is the
  first thing gate C must measure. Bounds, not measurements.
