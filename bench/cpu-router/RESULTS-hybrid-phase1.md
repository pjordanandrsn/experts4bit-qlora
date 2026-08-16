# RESULTS — Hybrid tier Phase 1: CPU router, gate G1

Engine: `experts4bit_qlora/engines/cpu_router.py` at the commit carrying this
file. Bench: `bench/bench_cpu_router.py`. Gate criteria fixed in advance by
the directive: nsys-verified zero blocking D2H syncs in the decode critical
path; per-layer round trip p50 ≤35 µs, p99 ≤100 µs at batch 1.

## Verdict

| G1 criterion | result | verdict |
|---|---|---|
| zero blocking D2H syncs in steady-state decode | 32 total (one per layer, warmup only) vs legacy's 6,400 (2 per layer-step) | **PASS** |
| round trip p99 ≤100 µs | **63.6 µs** | **PASS** |
| round trip p50 ≤35 µs | **45.1 µs** (42–46 across 4 runs) | **MISS** (+29%) |

A miss is reported exactly like a pass. The p50 gap is dispatch overhead
with a named, already-scheduled fix (below); the structural property the
phase exists for — routing with no host synchronization — holds.

## Box

Rented whole-machine slice: single-socket Zen 5 EPYC (48 vcpus exposed),
1× RTX 5090, PCIe Gen5 x16 (8 KiB pinned copy measured 2.51 µs H2D /
2.35 µs D2H on this box), torch 2.8.0+cu128, `OPENBLAS_NUM_THREADS=1`,
router thread under `taskset`. Destroyed after the run. (Different unit
than the Phase-0 blob box, same class; link latency measured in-session.)

## Arms

- **replica** (the gate instrument): 32 synthetic decoder layers at OLMoE
  geometry (E=64, H=2048, k=8) — attention stand-in → patched router →
  expert stand-in that consumes the device index tensor. Everything except
  the router is sync-free by construction, so the nsys table attributes
  cleanly. `--audit` mode removes all harness syncs from the token loop.
- **legacy**: identical layers, reference GPU router + `ids.cpu().tolist()`
  consume — the #105/#108 stall class, measured side by side.
- **model**: real OLMoE-1B-7B decode via transformers, 16/16 routers
  patched, greedy generation coherent (routing correct in vivo).

## The sync audit (nsys, 100 tokens × 32 layers, no harness syncs)

| cuda API | replica (CPU router) | legacy |
|---|---|---|
| cudaStreamSynchronize | **32** (warmup, one/layer; 0 steady-state) | **6,400** (2/layer-step) |
| cudaEventSynchronize | 0 (spin-poll never fell back) | 0 |
| cudaEventQuery (non-blocking) | 6,405 | 0 |
| cudaStreamWaitEvent (device-side) | 6,400 | 0 |
| cudaMemcpyAsync | 6,432 (D2H hidden + fused H2D idx/wts) | 6,400 |

Traces committed: `g1_audit_replica.nsys-rep`, `g1_audit_legacy.nsys-rep`
(+ `g1_nsys_api_*.txt` extracts). This is the per-layer contract from the
design: one D2H, one fused H2D, all waits event-scoped or device-side.

## Round trip and its dissection (replica, batch 1)

trip (CUDA events, D2H start → indices landed on device): **p50 45.1 µs,
p99 63.6 µs**. Host segment profile: wake 12.4 µs (launch latency +
attention tail — dependency time, not router work; drops to ~1.3 µs in the
deep-queued real decode), math 20.9 µs (~14 numpy/torch dispatches: gemv,
stable argsort, softmax algebra, pinned writes), push 3.9 µs, plus ~8 µs of
H2D submission + event overhead.

What the optimization rounds established (each measured, three wrong
theories killed by profiling): op-count trimming inside python plateaus at
~21 µs because per-dispatch cost dominates arithmetic at E=64; BLAS
threading and cold-weight streaming were not the binding term; a
cross-dtype D2H costs a hidden cast-kernel submission (~14 µs) — staging
must stay in model dtype with the exact bf16→fp32 zero-extend done in
numpy.

**The named fix:** Phase 2 brings a native C module (the AVX-512 kernel
build). The router epilogue becomes one C call (~3–5 µs replacing ~21 µs of
interpreter dispatch), projecting p50 ≈ 25–30 µs. G1's p50 gets re-measured
when that lands; the miss is carried, not waived.

## Honest notes

- **Wall-clock at replica scale favors legacy** (107 µs/layer vs ~150):
  with toy experts there is nothing for the eliminated sync to serialize
  against. The replica is a latency/structure instrument. The throughput
  claim belongs to Phase 3, where a 45 µs asynchronous round trip overlaps
  a ~2 ms expert step that the legacy pattern would serialize.
- Model arm: trip p50 reads 85 µs but that metric is confounded there —
  the GPU-side clock starts when the D2H fires (early, GPU deep-queued)
  while python arrives ~30–40 µs later; the host segment (54 µs) is the
  honest add. Math ran 47 µs in-model vs 21 in-replica; chain-warming the
  next layer's router weights did not close it — attribution open, tracked
  for the Phase-2 re-measure rather than guessed at.
- `router_logits` returns as `None` on the served path (all three
  supported blocks discard it in inference; grad/training falls back to
  the reference and gets real logits). Documented API deviation.
- Deterministic top-k: stable descending sort, ties to the lower expert
  index, identical rule on every backend; `assert_every=N` cross-checks
  CPU routing against the GPU reference and raises beyond tie-break cases
  (positive control in tests proves the checker fires).
- Found upstream while running the model arm: transformers' grouped-mm
  chooser admits sm_120 but `torch._grouped_mm` accepts only CC 9.0 —
  crashes OLMoE on Blackwell; pinned `_experts_implementation="eager"` in
  the bench.

## Receipts

`g1_replica.json`, `g1_legacy.json`, `g1_model.json`, nsys traces + API
tables, all in this directory.
