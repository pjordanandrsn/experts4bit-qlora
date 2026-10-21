# `p156-5090-1`: K39's decode against today's select tree at Qwen3-30B-A3B's W16 shapes (GO)

One RTX 5090 (cc 12.0, 170 SMs, 32,607 MiB, driver 595.58.03, a **575 W** power limit read on the box before any
install, a 3,090 MHz maximum SM clock) on an AMD Ryzen Threadripper PRO 5995WX host (128 CPUs, 251 GiB RAM), machine
40959, the pin. Launched 2026-10-20T00:10:44Z; the bench ran 00:12:35Z to 00:20:19Z. Cost $0.151, at $0.8222/h.
- **Code:** e4b `101ef5d7` (the registration's merge, #1857); grouped-nf4-gemm main at `68bbefc4` (#569's merge, after
  v0.49.0; K39 unreleased). The tripwire checked the sha256 of `nf4_smallm.py`, `int4_b32.py` and `nf4_grouped.py` on
  the box.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0 (the image's).

**Before the bench:**
- The power floor and refusals passed. The guard fit before the install passed: 4,440 s left against the 3,875 s
  needed.
- The tripwire, the self-tests (`p156_bench.py` 7 cases, `p156_reduce.py` 39) and the premise (K25's row-exact GPU
  tests, 4 passed) passed.

**The bench** (`bench_p156.json`) ran rc 0 in 462.0 s, against the registered allowance of 1,600 s:
- **Routing:** S0's picks, the pinned `rr.json.pt` (`3c59e72f`) and slice (`3020dbce`). There were no tile faults.
- **Identity (step 0, 48 layers):** in each projection all 144 prmt plans ran, and every one was bit-identical to T (the
  select tree at today's plan). There were no errors: no plan refused for shared memory.
- **The sweep:** 8 selection steps. Compiling and first-calling the 288 prmt variants took 98.8 s of host CPU.
- **Confirmation:** the 64 held-out steps. T, T2, P and C1-C3, every arm bit-identical to T on every rep.
- **Compiled metadata, both projections:** T is 80 registers, 0 spills, 14,592 B shared memory; P is 64, 0, 6,400 B.
- **Nsight Compute (reported only):** `ncu` was present, and the host refused the counters (`ERR_NVGPUCTRPERM`). The probe
  recorded it, and no capture was made.
- **GPU state** (`gpu_phases.csv`): 2,400 MHz at 72 W as the bench started; 2,272 MHz at 480 W as it ended, with the
  clock-event reason 0x4 (software power cap). Per-arm clocks were not recorded: see `RESULTS-p156.md`'s caveat.

**Verdict: GO** (`verdict_p156.json`):

| arm | decode, plan (gate_up / down) | median T/arm over 64 reps | SE(median) | lower bound |
|---|---|---|---|---|
| T2 (self-pair) | tree, today's / today's | 0.9985 | 0.0004 | |
| **P** | **prmt, today's / today's** | **1.1755** | 0.0014 | **1.1728** >= 1.05 |
| C1 (reported) | prmt, 64/256/4/2 / 128/128/8/3 | 1.9019 | 0.0019 | 1.8981 |
| C2 (reported) | prmt, 64/256/4/3 / 32/256/4/2 | 1.8313 | 0.0017 | 1.8279 |
| C3 (reported) | prmt, 64/256/8/3 / 128/128/4/2 | 1.8652 | 0.0013 | 1.8625 |

Plans read BLOCK_N/KC/warps/stages; today's is 32/64/4/3. T ran 12.872 ms a step for both GEMMs over 48 layers, and P
10.962 ms.

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store:** the launcher's receipt and ledger row, the instance and guard files,
  the teardown proof, which names the provider's instance, and the two files that carry the GPU's UUID, which no
  reading depends on (`p156/forensics.txt`, whose other facts are quoted above, and the runner's console log
  `p156/outer.log`). `STORE-ONLY.sha256` lists those three by their paths within the run's receipt.
- **Not copied:**
  - the staged scripts and the router replay, each byte-identical to the file `staged-p156.sha256` pins at `101ef5d7`
    (checked against the fetched copies);
  - the run's nonce markers and pip constraints file.
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.

Re-derive: `python bench/p156/p156_reduce.py --bench bench/p156/receipts/p156-5090-1/bench_p156.json --out {out}` writes `bench/p156/receipts/p156-5090-1/verdict_p156.json`
