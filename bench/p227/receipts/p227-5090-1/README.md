# `p227-5090-1`: the per-row GEMV against K25 at 2 to 8 decode tokens (Mixtral PASS, Qwen3 MISS)

One RTX 5090 (cc 12.0, 170 SMs, driver 580.126.18, a **575 W** power limit read on the box before any install) on an AMD
EPYC 9334 host (128 CPUs, 755 GiB RAM), 2026-10-23T13:26:11Z to 13:38:47Z, through the guarded launcher. Cost $0.083.
- **Code:** e4b `b7146363` (the registration with Amendments 1-4, e4b#2036's merge); grouped-nf4-gemm v0.50.0
  `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's), transformers 5.17.0.

**Before the bench:**
- The power floor, the refusals and the pre-install fit passed (4,143 s left against the 3,375 s needed).
- The self-tests passed (`p227_bench.py` 15 checks, `p227_reduce.py` 45), and so did the premise (25 passed, as
  registered: e4b#2009's GPU contract, 20, and the served T == 1 route's tests, 5).

**The bench** (`bench_p227.json`) ran rc 0 in 316.1 s, with 22,823 clock samples and no sampler error.

**Verdict: `qwen3=MISS mixtral=PASS`** (`verdict_p227.json`):

| family | T = 2: B/K upper bound | T = 4: B/K upper bound | verdict | B2 trigger (B / floor at T = 8) |
|---|---|---|---|---|
| Mixtral-8x7B | 0.3601 | 0.4655 | **PASS** (both <= 0.70) | 2.00 (lower bound 1.97): fires |
| Qwen3-30B-A3B | 0.7478 | 0.9047 | **MISS** | 1.59 (lower bound 1.57): fires |

Clock-adjusted bounds equal the raw ones (no arm ran a faster SM clock). Self-pairs 0.9999 to 1.0005; B against K within
1.5e-2 everywhere (largest 0.0078). The full table is in [../../RESULTS-p227.md](../../RESULTS-p227.md).

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store** (`STORE-ONLY.sha256`, by path within the run's receipt): the
  launcher's teardown proof and instance record, which name the provider's instance. The launcher's receipt, its ledger
  row and the guard files stay there too.
- **Not copied:** the staged scripts, each byte-identical to the file `staged-p227.sha256` pins at `b7146363`; the nonce
  markers and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
