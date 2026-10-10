# DQ11 draw-4 measured supplement

Measured-private evidence from 2026-10-10, at reviewed #1613 merge
`e672f98478295b86ccee01e4500a8a83e41f33f4`. The fresh real initial read ran
first, then the seventh full Tiny rehearsal, under one sole A2000 claim and
the same absolute deadline `1791674831`, with a 300-second teardown reserve.
Both containers exited 0 without OOM. The exact idle baseline, 1526 MiB and
0% utilization, was restored before release
`20261010T215428Z-71535-95c2d9104353`. Rental cost was $0.

## Real Mistral initial comparison

The original scorer and strict frozen-value equality passed for L/U/U0 on
the pinned real model and canonical inputs/448 FP32 adapters. U/U0 restored
Unsloth's zeroed padding row from the checkpoint. Both named the deviation,
and their scored token-0 counts were zero on each text. There were zero
optimizer updates. The unchanged absolute 0.05 PPL budget passed:

| Text | L PPL | U PPL | Absolute delta PPL | U0 minus U |
| --- | ---: | ---: | ---: | ---: |
| Alpaca heldout | 2.52318402155865 | 2.5231012284736565 | 0.00008279308499359317 | 0 |
| WikiText test | 5.103491883488553 | 5.1003000466336665 | 0.003191836854886887 | 0 |

Each text had 16,376 shifted targets. U0/U also agreed exactly in NLL.
The original report was re-derived on CPU, then independently by the
maintainer (`20261010T214801Z-54181-3c8fdee821f7`). The previous `3daa0747`
frozen-embedding refusal remains immutable; no corrected copy supplied
this result. This addresses Amendment 5's unmeasured initial arithmetic
risk only on A2000/cu128/sm86; trained and 5090/cu130 quality remain unmeasured.

| Real initial phase | Elapsed seconds |
| --- | ---: |
| Git prerequisite | 10.186 |
| Create venv | 3.320 |
| Bootstrap | 176.276 |
| Prepare real model | 117.706 |
| Initial L | 162.574 |
| Initial U | 173.935 |
| Primary report | 6.440 |
| Optional initial U0 | 133.335 |
| Optional U0 report | 5.731 |

The phase sum was 789.503360 s; the ledger wall span was 789.506234 s.
After the primary report, 6252.605477 s remained beyond the reserve;
after the optional U0 report, 6113.538304 s remained.

## Seventh complete Tiny instrument

All 16 required phases passed: fresh bootstrap/preparation, three C proofs,
initial correctness, six reads in L/U/U0/U0/U/L order and final correctness.
Required elapsed time summed to **521.889082 s** (about 522 s); wall span
was 521.908010 s. At final required completion, 5401.440719 s remained
beyond the reserve. The three optional shipped spreads ran last and
completed; after them, 5333.565583 s remained. The deadline was never reset
between the real and Tiny runs.

CPU review re-derived all six receipts, proof metadata, policies, canonical
inputs/initializer, warnings, deadline/min-cap arithmetic and all 12 raw
gradient files and their pairwise comparisons. Eight U/U0 proof/read/spread
receipts recorded row-0 restoration and zero/zero scored token-0 counts.
The observer/clean proof tensors were not retained; their same-arm bitwise
metadata was checked, with separate optional raw gradient pairs re-derived.
The maintainer independently reran `dq11_rehearsal.py check` on the raw
receipts and accepted the full path
(`20261010T220329Z-86612-a0d6caaac7d0`); his check omitted optional gradient
packets, which the CPU audit covers.

## Immutable private handoffs and limit

- Real index: `adertha://experts4bit-qlora%231083/reviews/dq11-real-quality-e672f984-index.json`, SHA-256 `de18f2c44179847d017136187f09647741690c105e749c17439179b31f86acaf`.
- Seventh index: `adertha://experts4bit-qlora%231083/reviews/dq11-seventh-e672f984-index.json`, SHA-256 `382d58a2535d80940cd34812eee1ba38500c27bbb3cad0f398ed6a254644a9ea`.

Both indexes and their review packets passed fresh hash-bound readback.
They retain source/nonce, raw ledgers, logs, scores and independent checks.
Both runs are `science_eligible=false`. Tiny timing cannot predict full
Mistral/cu130/SM120 completion within two hours. Neither run licenses speed,
capacity, shipping or scientific recommendations. A paid run that exhausts
its unchanged required-phase deadline remains VOID. This supplement grants
no quote or rental authorization; see [Amendment 6](DQ11-AMENDMENT-6.md).
