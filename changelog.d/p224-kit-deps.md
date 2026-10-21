### Fixed

- P224 kit: the runner installs Loggetta's other requirements (`datasets>=2.14`, `peft>=0.21.2`) under the held torch,
  since Loggetta itself goes in `--no-deps` to keep the pinned e4b and grouped-nf4-gemm, and the tripwire imports the
  training path before any fetch. Without it every Loggetta train would fail at import on the box. Found before any
  run; no receipt is affected.
