### Added

- P235 results (`bench/p235/RESULTS-p235.md`). `p235-5090-1`, the one reading of PREREG-p235.md at e4b `7be9e610`,
  re-read P58's same-box vLLM head-to-head at the release pins: experts4bit-qlora 0.58.0 and grouped-nf4-gemm 0.51.0
  from PyPI, each distribution proven byte-identical to its published wheel. vLLM 0.32.0 / e4b int4 = 1.038 at B=1 and
  1.203 at B=16 (vLLM ahead); vLLM 0.30.0 / e4b int4 = 1.031 and 1.187; every arm VALID, every self-pair inside 1.03x.
  Quoted beside P58's 1.087 / 1.396, measured on another box, so the difference is not read as an e4b gain. One RTX 5090 at a
  470 W limit (recorded; the ratio is inside one box). It licenses nothing and moves no default. $0.86. The receipt copy
  is under `bench/p235/receipts/p235-5090-1/`.
