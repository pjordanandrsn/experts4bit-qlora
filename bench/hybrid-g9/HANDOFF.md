# Hybrid CPU/GPU tier — campaign handoff (2026-08-19)

State of the Stage-2 program after the 2026-08-18/19 measurement
campaign. Everything below is merged to main in both repos
(grouped-nf4-gemm and experts4bit-qlora); every number has receipts in
`bench/hybrid-g9/<round>/`; every box was destroyed with SSH-refused +
API-null verification.

## Gate scoreboard

| gate | clause | status |
|---|---|---|
| G0-G5 | Stage 1 (calibration, router, kernels, executor, prefetch, backward) | CLOSED (see Stage-1 receipts) |
| G6 | tiered paged KV | CLOSED |
| G7 | FP8 KV quality/bandwidth/kernel | CLOSED 3/3 |
| G8 | balance >= 0.80, B=8 | **CLOSED — 0.978 reference host; 0.87-0.93 on two other classes** |
| G8 | balance >= 0.80, B=16 | **OPEN — best clean 0.698 (Zen 5); needs ~-14% on the wall** |
| G9 | decode degradation <= 20% | CLOSED where serving lives (5.0% @chunk512, decode-dominated) |
| G9 | 140 tok/s aggregate on gpt-oss-120b | OPEN — 45.4 best (Qwen3-30B stand-in; gpt-oss blocked, see below) |

Campaign arc: G8 B=8 went 0.124 -> 0.978; G9 went 23.9 -> 45.4 tok/s
with TTFT 12-17 s -> 4.9 s warm.

## Mechanisms landed (all merged, each with a receipts round)

placement concentration via solver cost constants (fixed=55/per_row=2)
· dequant hoist (AVX-512 cells) · three KV stream-sync removals ·
torch-intraop cap (12x on big-core hosts) · thin-layer routing
(offload_thin_uniq) · pool subset engagement (+ gen-packed race fix) ·
work-stealing inside jobs · adjustable spin window (pool_spin_us) ·
fused expert-FFN kernel (exact, opt-in; default-False on serving
measurements) · warm-both-shapes gate discipline.

## The serving playbook (current best configuration)

pool = min(32, physical cores); `torch.set_num_threads(8)` whenever the
pool is engaged; `pool_spin_us(1500)`; solver `cpu_us_fixed=55,
cpu_us_per_row=2` at batch; `offload_thin_uniq=4` (8 helps B=16);
`fused_ffn=False`; warm one prefill chunk + one decode step before any
TTFT clock.

## Instrument laws (violations produced wrong conclusions this campaign)

1. Warm both shapes before TTFT clocks (triton compile lands in the
   first run's window otherwise).
2. Pin torch intraop when the pool runs (default = physical cores and
   it thrashes the pinned workers).
3. Tee stage outputs to files — `| grep` swallowed two crashes.
4. Probe a claimed-dead box directly before re-renting (the `ports`
   field never populates on proxy ssh; five rentals were burned on a
   fabricated signature).
5. Compare kernel numbers only at matched call shapes (cross-shape
   extrapolation manufactured a phantom 2x executor loss).
6. Single-run validations whose arm ran last in a process sequence are
   confounded until re-run in A/B/A order.
7. A balance ratio moves with the GPU side: faster GPU raises the bar
   for the DRAM wall. State the host class with every number.

## Open work, ordered

1. **G8 B=16 (~-14%)**: (a) cache-block columns against row chunks in
   the cells (the receipts' rows-at-equal-bytes signature is the
   fingerprint of per-column activation re-streaming); (b) row-aware
   per-call GPU routing at B>=16; (c) re-derive constants at B=16
   shapes. See `b16close/RESULTS-b16.md`.
2. **G9 engine work** (Phase-10-class): non-expert step time (~120
   ms/step at B=8 x 48 layers: attention kernels + python dispatch) and
   batched prefill (serial `ids[None]` chunk forwards bound TTFT at
   ~1.1 s x queue depth).
3. **gpt-oss-120b arena bias baking** — the G9 headline model refuses
   arena serving until per-expert biases ride the arena (loader
   deliberately refuses rather than dropping them).
4. **MXFP4 hybrid DRAM tier** — segments swap in `_HybridTier` + the
   existing mxfp4 CPU kernel; currently NF4-only.
5. **fp8-COMPUTE model-quality certification** (owed at G9; kernel
   mode exists and is fast, `compute="f32"` stays default until then).
6. **Bare-metal profiling session** (Latitude grants perf counters) for
   the intracall interference term (in-context dn at 5-6x isolated).
7. Deferred: per-worker futexes (wake-subset beyond join-subset);
   Phases 10-11 proper (demotion/park of FP8 blocks; see the Stage-2
   directive).
8. NO releases tagged — tagging stays on the owner's explicit word.

## Where things are

Receipts: `bench/hybrid-g9/{box,fixbox,remeasure,coresweep,gate1,gate2,
final,intracall,spinwindow,barconfirm,b16axis,b16close}/` — each with a
RESULTS-*.md that states what bound and what did not. Harnesses:
`bench/hybrid-g9/step_decomp.py` (in-repo); box scripts in the session
scratchpads are disposable and reconstructible from the RESULTS files.
Architecture: `docs/hybrid/ARCHITECTURE-NOTES.md` (this file is linked
from there).
