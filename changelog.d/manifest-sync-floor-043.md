### Changed

- `docs/system-manifest.json` is synced byte for byte with grouped-nf4-gemm's: a compatibility record for this package
  0.55.0 and later floors the `fast` extra at grouped-nf4-gemm 0.43.0 (P116's registered consequence, the release that
  makes `GNF4_GEMV_BW=auto` the default), and the previous record now covers 0.35.0 to 0.54.x. Nothing installs
  differently yet: 0.54.x keeps its 0.30.0 floor, and 0.55.0 raises it.
