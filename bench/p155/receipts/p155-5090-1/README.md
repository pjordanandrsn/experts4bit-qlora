# `p155-5090-1`: K25's plan at Qwen3-30B-A3B's W16 shapes (NO_GO)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.91.07, a **575 W** power limit read on the box before any install) on an AMD
EPYC 7K62 host (96 CPUs, 503 GiB RAM), 2026-10-19T22:26:16Z to 22:44:14Z. Cost $0.274, at $0.8778/h.
- **Code:** e4b `4d3fd170` (the registration's merge, #1851); grouped-nf4-gemm v0.48.0 `ef2faf38`, whose `nf4_smallm.py`
  and `int4_b32.py` are byte-identical to v0.47.1's (the tripwire checked both sha256s on the box).
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's).
- **The same card as P152's and P154's reads:** the GPU UUID in `forensics.txt` is the one in theirs.

**Before the bench:**
- The power floor and refusals passed. The guard fit before the install passed: 5315 s left against the 4937 s needed.
- The tripwire, the self-tests (`p155_bench.py` 5 cases, `p155_reduce.py` 30) and the premise (K25's row-exact GPU tests,
  4 passed) passed.

**The bench** (`bench_p155.json`) ran rc 0 in 895.6 s, against the registered allowance of 2450 s:
- **Routing:** S0's picks, the pinned `rr.json.pt` (`3c59e72f`) and slice (`3020dbce`). There were no tile faults: the real
  tiles were a prefix, one per routed expert, on every table built.
- **Identity (step 0, 48 layers):** in each projection 126 of the 144 plans ran, and every one was bit-identical to today's
  plan. The other 18 refused with `OutOfResources`, shared memory over sm_120's 101,376 B per block.
- **Selection:** 8 steps. Compiling and first-calling the 252 variants took 169.2 s of host CPU, at most 2.4 s for one.
- **Confirmation:** the 64 held-out steps. T, T2, C1-C3 and K, every arm bit-identical to T on every rep.
- **Compiled metadata:** today's plan is 80 registers, 0 spills, 14,592 B shared memory, in both projections.
- **Nsight Compute (reported only):** `ncu` was present, and the host refused the counters (`ERR_NVGPUCTRPERM`). The probe
  recorded it, and no capture was made.

**Verdict: NO_GO** (`verdict_p155.json`):

| arm | plan (gate_up / down) | median T/arm over 64 reps | SE(median) | lower bound |
|---|---|---|---|---|
| T2 (self-pair) | today's / today's | 0.9999 | 0.0001 | |
| **C1** | 32/128/4/2 / 64/64/8/1 | **1.0188** | 0.0004 | **1.0180** < 1.15 |
| C2 | 32/128/4/1 / 32/64/4/1 | 1.0098 | 0.0005 | 1.0089 |
| C3 | 32/128/4/3 / 16/64/2/1 | 0.9954 | 0.0004 | 0.9946 |
| K (compaction, reported) | today's, real tiles only | 1.0032 | 0.0001 | |

Plans read BLOCK_N/KC/warps/stages. Today's plan ran 12.075 ms a step for both GEMMs over 48 layers.

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store:** the launcher's receipt and ledger row, the instance and guard files,
  and the teardown proof, which names the provider's instance. `STORE-ONLY.sha256` lists the teardown proof by its path
  within the run's receipt.
- **Not copied:**
  - the staged scripts and the router replay, each byte-identical to the file `staged-p155.sha256` pins at `4d3fd170`
    (checked against the fetched copies);
  - the run's nonce markers and pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name or an endpoint.
