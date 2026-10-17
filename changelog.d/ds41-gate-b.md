### Added

- `bench/ds41-gate-b/`: gate B for DeepSeek-V4.1-Flash-1017, an analytic memory, residency and transfer model ($0; no weights,
  no build). Full host residency needs about 501 GB of host RAM; Q235's 239 GB-floor boxes hold the 203 GB Engram tables but
  stream the 289 GB of experts from disk. Decode transfer bounds at batch 1 and 128K: 0.43 tok/s on the A2000 host, 1.55–2.29 on
  the 239 GB floor, 4.85–6.86 with everything in RAM at gen-4 H2D. Bounds, not measurements.
