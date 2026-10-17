### Added

- An analytic note on Q235 Shape B's staged bytes (`bench/q235/staged-bytes/`), from records only. A Shape B pass stages
  M × (2n − 1) whole layers: 748, or 1,016.5 GB at 94 layers. Layer-major accumulation would stage 187 (−75 %), and staging only
  each micro-batch's routed experts 0.734 of them (measured routing, −27 %); together −78 %. Neither may run in the registered box
  without its own amendment. Documentation only.
