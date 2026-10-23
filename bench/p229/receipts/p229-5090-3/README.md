# `p229-5090-3`: the int8 lm_head at 2, 4 and 8 rows on K37 (GO_ROWS_8)

One RTX 5090 (cc 12.0, driver 580.105.08, a **575 W** power limit read on the box before any install) on an AMD EPYC
7713 host (256 CPUs, 440 GiB RAM), 2026-10-23T18:34:32Z to 18:37:19Z, through the guarded launcher. Cost $0.061.
- **Code:** e4b `32fda64d` (the registration with Amendments 1-6, e4b#2061's merge); grouped-nf4-gemm v0.50.0
  `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's).

**Before the bench:**
- The power floor, the refusals and the pre-install fit passed (3,524 s left against the 2,625 s needed).
- The self-tests passed (`p229_bench.py` 8 checks, `p229_reduce.py` 25), and so did the premise, exactly as registered:
  the real kernels on the card, 11 passed (`logs/premise_gpu.log`), and the route's CPU suite, 45 passed
  (`logs/premise_cpu.log`).

**The bench** (`bench_p229.json`) ran rc 0 in 7.6 s, with 933 clock samples and no sampler error. The installed route was
as registered: `rows_max` 8, K37 `dense_i8.linear_i8`, plan (32, 256, 4) at every M from 2 to 8. The engagement was as
registered: K36 at one row, K37 at 2, 4 and 8.

**Verdict: `GO_ROWS_8`** (`verdict_p229.json`):

| M | I/D median | I/D upper bound | I/K median | I/K upper bound | rules |
|---|---|---|---|---|---|
| 2 | 0.5109 | **0.5109** | 1.0217 | **1.0217** | PASS |
| 4 | 0.5136 | **0.5148** | 1.0272 | **1.0296** | PASS |
| 8 | 0.5081 | **0.5085** | 1.0108 | **1.0116** | PASS |

Bars: I/D <= 0.70 and I/K <= 1.25, both on the upper bound and on its clock-adjusted value. Clock-adjusted bounds equal
the raw ones by construction: clocks are sampled per M window, so the arms within an M share one clock median. I against
D within 1.5e-2 in every rep (largest 0.0078). The full table is in [../../RESULTS-p229.md](../../RESULTS-p229.md).

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store** (`STORE-ONLY.sha256`, by path within the run's receipt): the
  launcher's teardown proof and instance record, which name the provider's instance. The launcher's receipt, its ledger
  row and the guard files stay there too.
- **Not copied:** the staged scripts and staged tests, each byte-identical to the file `staged-p229.sha256` pins at
  `32fda64d`; the nonce markers, the run id and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
