### Read: P114 — on one rented RTX 5090, `bnb.matmul_4bit` costs 1.6–2.0× native bf16's energy per op; the 5090 row supersedes both A2000 energy rows (READ; bench, docs and register)

- **The run.** `p114-5090-1`: one RTX 5090 at a 500 W power limit (driver 570.133.07), bitsandbytes 0.50.2, the two
  energy harnesses unchanged, three passes each. It cost $0.148, with teardown proven.
  - `bench/p114/p114_reduce.py` reads **READ**: every spread is ≤ 0.047 against the registered 0.10.
  - The reducer reproduces the verdict byte-identically from the committed receipts (`bench/p114/RESULTS-p114.md`,
    Reproduce).
- **The reading** (medians of three passes). Total J/op over native bf16 on one OLMoE-dims projection:

  | workload | `bnb.matmul_4bit` | dequantize → `linear` |
  |---|---:|---:|
  | decode | 1.748 | 3.293 |
  | prefill | 1.601 | 1.539 |
  | train | 1.965 | 1.405 |

  The fused 4-bit MoE forward's J/token at batch 4096 is 0.063 of batch 64's (≈16×).
- **The registered prediction that `matmul_4bit` would read near 1× at decode is refuted: 1.748×.** It draws half
  native's power at 0.28 of native's op rate. The other directions held.
- **Register.** `e4b.train.energy-honest.5090.2026-10-05` (measured) supersedes
  `e4b.train.energy-honest.scoped-a2000` and `e4b.train.energy-honest.a2000-bnb0502.2026-10-04`, and
  `e4b.train.energy-honest` now points at it directly. The A2000 rows stand as measured on their card.
- **Quotes.**
  - README, STATUS, SOLUTIONS, `capabilities.json`, the bitsandbytes solution page, BITSANDBYTES.md and
    STORAGE-MODES.md cite the new row with its scope. Where a figure is quoted, the dequant arm keeps its own numbers.
  - METHODOLOGY §10 gains the 5090 tables (c), and the A2000 tables stay as the record.
