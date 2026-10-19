# `p152-5090-2`: the census (READ)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.91.07, a **575 W** power limit read on the box before any install) on an AMD
EPYC 7K62 host (96 CPUs, 503 GiB RAM), 2026-10-19T16:36:54Z to 16:59:55Z. Cost $0.506, at $0.9037/h including 200 GB.
- **Code:** e4b `334ef5b5` (the registration's merge, #1815), grouped-nf4-gemm v0.47.1 `38b30888`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2, Nsight Systems 2025.6.1.190;
  Qwen/Qwen3-30B-A3B @ `ad44e777`.

The power floor, install, tripwire and self-tests passed: P136's reducer, `p152_reduce.py` 20 cases, `pername_table.py`
10, P136's box 5 and P136's 19 premise tests. Both speed arms, the four captures (W1 and W16, graph and node mode) and
the routing pass completed with rc 0.
- **The server:** the shipped default at 16 slots x 4096 tokens, graphs on, every knob unset. Each arm recorded
  `fuse_qkv_n` 48, `fuse_t1_glue_n` 193, `fuse_t1_glue_r2_n` [48, 48] and `fuse_router_epilogue_n` 48.
- **The denominators** (unprofiled): W1 **3.891 ms** (S1a 3.8913, S1b 3.8908), W16 **16.4969 ms** (16.5073, 16.4864).
  Self-pairs W1 1.0001, W16 1.0013.
- **The census:** both tables kept 61 steps and closed every replay exactly (`closure_ns_max` 0). Profiled periods are
  context only: W1 4.804 ms, W16 17.528 ms.
- **The routing pass:** 64 steps x 48 layers recorded, with two route faults (see below), so the W16 expert floor reads
  **UNREAD** by the registered rule.
- **GPU state** (`gpu_phases.csv`, sampled at each phase's start and end): SM clocks 180 MHz at the first edge (idle,
  before the first load), then 2392-2400 MHz at every other edge; no clock event reason set at any edge. Reported only.

**Verdict: READ** (`verdict_p152.json`). Named W16 candidates, by removable time over the unprofiled W16 step:

| | line | launches/step | removable ms | share |
|---|---|---|---|---|
| 1 | `_gemm_nf4_grouped_smallm` 136x48 (gate_up) | 48 | 8.3951 | 50.89 % |
| 2 | `_gemm_nf4_grouped_smallm` 136x64 (down) | 48 | 3.3530 | 20.32 % |
| 3 | `Kernel2` 8x16 (dense) | 48 | 1.1111 | 6.73 % |
| 4 | `Kernel2` 8x40 (dense) | 48 | 0.7462 | 4.52 % |
| 5 | `_fp8_paged_decode_split_f8dot` 320 CTAs (attention) | 48 | 0.6586 | 3.99 % |

**The routing record.** `route_b16.json` records `E4B_INT4_PREFILL` resolving to `k19`, K19's mode `auto`, both grouping
flags off, **0** uniform-int4 expert stores on 48 MoE layers, and 16 rows grouping as (singleton, device) = (False,
False). The served expert store is NF4, not int4, so the registered K19 route premise did not hold. The pass's routing
statistics are reported as recorded: a mean of 57.02 unique experts per layer-step (40 to 81; uniform routing would
give 82.42), 2736.8 a step, and a hottest expert holding 11.0 % of a layer's hits.

The launcher's receipt and ledger row are in the private receipt store, and the teardown is proven there. `SHA256SUMS`
covers every file here. Two kinds of file stay only in that store, byte-exact, and `STORE-ONLY.sha256` lists them by
their path within the run's receipt:
- fourteen logs that carry carriage returns from the box's progress output;
- the run's instance file and the launcher's teardown proof, which name the provider's instance.

The staged scripts are not copied: each is byte-identical to the file `staged-p152.sha256` pins at `334ef5b5`.
`calib.json` is not copied either: it is the kit's frozen calibration file, not a reading from this box.
