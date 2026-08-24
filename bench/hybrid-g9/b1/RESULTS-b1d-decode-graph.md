# RESULTS — B1d device-driven decode loop: CERTIFIED-WITH-OPEN-MECHANISM

Run 2026-08-24 against `PREREG-b1d-decode-graph.md` (#227) +
`AMENDMENT-b1d-capture.md`, on an EPYC 9655 + RTX 5090 (vast, NUMA
pre-gate 178.7 GB/s — one earlier slice was auto-refused at 138.8;
destroyed + verified zero). Receipts in `receipts-b1d/`; verdict by
`b1d_verdict.py` (self-tested, 8 branches).

## Stage A (capture smoke + identity)

Three capture failures, each named and treated, receipts kept:
1. Timing CUDA events in the attention wrapper — capture-illegal;
   both arms unwrap (instrument, not engine).
2. Allocator sync under VRAM pressure + stray-thread context probes —
   `empty_cache()` before capture, side-stream warmup,
   `thread_local` capture mode.
3. **The structural one** (`CUDA_LOG_FILE` + traceback):
   `unique_consecutive` in `_fused_over_stack` syncs for host group
   sizes — data-dependent grids cannot capture. AMENDMENT-b1d-capture
   registered the singleton-groups path BEFORE it was built: at T=1 a
   token's top-k ids are distinct, so `sizes=[1]*k` is a host CONSTANT
   and the ids ride as a device tensor; no sort, no unique, no unsort.
   Dispatch-algebra equality pinned in CI with duplicate-id fixtures
   (3 tests, mocked GEMM).

Smoke result: capture succeeded; **127/127 per-step logits hashes
bitwise-identical to eager, positions aligned** (the alignment gate
Bugbot asked for — capture-neutrality assert held: no state advance
during capture). Bonus finding: the singleton dispatch alone cut the
EAGER collapsed step 49.8 → ~43 ms (the T=1 sort/unique tax).

## Stage C (the cert)

| gate/bar | registered | measured | result |
|---|---|---|---|
| GS B=16 | certified band | 133.2 ms | PASS |
| G0 (eager/graph) | < 7.5% | 1.43% / **0.011%** | PASS |
| G1 | tokens bitwise | 127/127 | PASS |
| H-G | graph ≤ 20 ms | **15.19 ms** | PASS |
| H-D | occupancy ±15% | 12.55 vs ≤15.19 (proxy) | FAIL* |

**⇒ CERTIFIED-WITH-OPEN-MECHANISM: ship.** Single-stream all-resident
**65.8 tok/s (15.19 ms/step), 2.79× over the eager collapsed loop**,
bit-identical decode, B=16 untouched.

*The open mechanism, filed: the graph-side H-D input was the syncd
per-step median — an UPPER BOUND (sync + token-log copies included),
not measured occupancy; the eager side was profiler-measured (12.55).
Proxy-vs-real is not a like-for-like ratio. Closing it needs a
profiled replay window (kineto records graph-launched kernels); the
~2.6 ms gap decomposes into replay-launch overhead + log copies +
any real kernel delta. None of it moves H-G, which is wall-measured.

## The ladder after rung 2 (425 single-stream)

14.1 → 20.1 (collapse) → **65.8 tok/s (this cert)**. The step is now
~15 ms of essentially pure device work. Rung 3 — the M=1 kernel
roofline — owns the descent toward ~2.5-4 ms (measured ceiling
1573.9 GB/s, ~3.2 GB/token ⇒ ~480 tok/s roofline; the M=1 GEMV runs
at ~15% of it today). Rung 4 (speculative decoding) crosses 425.
