# `p228-5090-r1`: K1's pinned cuBLASLt GEMM for o_proj and the router at 2 to 8 rows (o_proj=PASS router=PASS)

One RTX 5090 (cc 12.0, 170 SMs, driver 580.126.18, a **575 W** power limit read on the box before any install) on an
AMD EPYC 9334 host (128 CPUs, 755 GiB RAM), 2026-10-23T22:46:35Z to 22:49:04Z, through the guarded launcher. Cost $0.04.
- **Code:** e4b `32fda64d` (the registration with Amendments 1-7, e4b#2061's merge), 0.57.0.
- **Stack:** torch 2.8.0+cu128, transformers 5.17.0, cuBLASLt 120804 (torch's own wheel).

**Before the bench:**
- The power floor, the refusals and the pre-install fit passed (3,489 s left against the 3,000 s needed).
- The tripwire passed (0.57.0, cuBLASLt 120804), and so did the self-tests (`p228_bench.py` 11 checks, `p228_reduce.py`
  26) and the premise, exactly as registered: K1's GPU and CPU suites on the card, **37 passed** (`logs/premise.log`).

**The bench** (`bench_p228.json`) ran rc 0 in 78.7 s (selection 10.5 s, reading 67.8 s), with 11,582 clock samples and
no sampler error. Block A selected algo 21, tile 5, split_k 16, reduction scheme 4, 20 stages in all six cells, and block
B timed that pin through e4b's installed path. The engagement was as registered: P counted 48 pinned `o_proj` calls and
384 pinned router calls at each M; D and T1 counted none.

**Verdict: `o_proj=PASS router=PASS`** (`verdict_p228.json`):

| site | M | P/D median | P/D upper bound | rules |
|---|---|---|---|---|
| `o_proj` | 2 | 0.5978 | **0.5981** | PASS |
| `o_proj` | 4 | 0.6008 | **0.6009** | PASS |
| `o_proj` | 8 | 0.6023 | 0.6025 | never rules |
| router | 2 | 0.2831 | **0.2832** | PASS |
| router | 4 | 0.2842 | **0.2843** | PASS |
| router | 8 | 0.8838 | 0.8841 | never rules |

Bar: P/D <= 0.85 at both M = 2 and M = 4, on the upper bound and on its clock-adjusted value. Every arm in a cell ran at
the same median SM clock, so the clock-adjusted bounds equal the raw ones. P against D within 1.5e-2 in every rep
(largest 0.0078). The router at M = 8 reads above the bar; M = 8 never rules, and
[../../RESULTS-p228.md](../../RESULTS-p228.md) gives the cause (the default goes split-K at 8 rows) and the full table.

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store** (`STORE-ONLY.sha256`, by path within the run's receipt): the
  launcher's teardown proof and instance record, which name the provider's instance. The launcher's receipt, its ledger
  row and the guard files stay there too.
- **Not copied:** the staged scripts and staged tests, each byte-identical to the file `staged-p228.sha256` pins at
  `32fda64d`; the nonce markers, the run id and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
