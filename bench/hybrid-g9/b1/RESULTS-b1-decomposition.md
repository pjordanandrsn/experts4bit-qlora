# RESULTS — B=1 single-stream decomposition: BRANCH-2 (abstraction tax)

Run 2026-08-24 against `PREREG-b1-decomposition.md` (#222), on the same
EPYC 9755 + RTX 5090 box as the T5b Phase A cycle (vast 48576023-era
box id 15, destroyed + verified zero). e4b at `a15d8e9` + the #223
mechanics/instrument fixes (measured tree = branch `b1-r1-mech`
@ `8d24316`). Receipts in `receipts-b1/`; verdict by `b1_verdict.py`
(self-tested, 10 branches). **No optimization was performed — the
decomposition and the branch decision are the deliverable.**

## The B scaling curve (one box, certified command line, amort off)

| B | ms/step | aggregate tok/s | per-stream tok/s |
|---|---|---|---|
| 1 | 70.9 | 14.1 | 14.1 |
| 2 | 81.1 | 24.7 | 12.3 |
| 4 | 96.9 | 41.3 | 10.3 |
| 8 | 113.5 | 70.5 | 8.8 |
| 16 | 141.0 | 113.5 | 7.1 |

The B=16 point passed the GS shape-gate (141.0 ms, attn 46.0) — the
box is at the certified operating point. Halving B never halves the
step: the host-dominated step has a large B-independent component,
which is exactly what the three arms then decompose.

## H / R0 / R1 at B=1 (the causal decomposition)

| arm | step ms | tok/s | A/A spread | what it removes |
|---|---|---|---|---|
| H (production hybrid) | 70.9 | 14.1 | 1.05% | — |
| R0 (all-VRAM, executor intact) | 66.2 | 15.1 | 0.61% | physical heterogeneity |
| R1 (pipelined, all experts hot) | **49.4** | **20.2** | 0.11% | the hybrid machinery |

- **Residency tax (H→R0): 6.6%.** The CPU/DRAM tier is NOT what makes
  B=1 slow at this placement — the solver's VRAM set covers B=1
  routing well (H even decoded the same 128 tokens as the all-GPU
  arms: the DRAM tier's rounding never surfaced).
- **Abstraction tax (R0→R1): 25.4%** — above the preregistered 25%
  threshold. The hybrid executor's per-layer machinery (tier
  splitting, cold-path checks, bookkeeping, pool presence) costs a
  quarter of the all-resident step even when NOTHING is offloaded.
- **Identity (G1): R0 ≡ R1 bit-identical over all 128 tokens** —
  across two DIFFERENT executors (hybrid all-hot vs pipelined
  device-id GEMV), a strong equality, now measured for real (see the
  erratum below). H agreed 128/128 as well. Per-arm determinism pairs
  identical; the H stats run (amort on) decoded identically to the
  timed run, binding the routing statistics to the timed arm.

## Attribution (profiled runs; #216 occupancy convention)

Real kernel occupancy per step: H 12.1 ms, R0 13.7 ms, **R1 13.3 ms**
— essentially constant across arms (the device does the same work; the
arms differ in host). R1's clean step spends **3.7× its kernel time in
host work** (49.4 vs 13.3), so the fast anchor (≤2×) reads SLOW: even
the narrowest existing resident path is host-bound at M=1.

Host brackets at B=1 (profiled steps — instrument-inflated, but
internally comparable):

| region ms/step | H (88.2) | R0 (88.3) | R1 (63.1) |
|---|---|---|---|
| attention host | 21.7 | 22.0 | 20.5 |
| MoE experts host | **34.0** | **32.8** | **12.8** |
| router/top-k | 5.3 | 5.4 | 5.0 |
| residual (dense/norms/glue) | 32.0 | 32.9 | 29.3 |

The abstraction tax localizes: attention, router and residual are
near-identical across arms, and the entire R0→R1 gap sits in the
EXPERTS DISPATCH bracket (32.8 → 12.8 ms). The collapse fast path has
a named address. The ~30 ms residual is the B-independent
dense/norm/glue floor the scaling curve predicted.

## Routing locality (H series, the prefetch evidence bar)

Median across layers, token-to-token: Jaccard **0.33**, repeat
probability **0.50**, ~8 uniques/layer/step at k=8. Half the routed
set repeats step-over-step — moderate predictability. Any future
prefetch registration must clear this trace's numbers; nothing here
licenses speculative prefetch by itself.

## Preregistered decision

**BRANCH-2 — abstraction tax: implement the all-resident collapse fast
path.** When the placement is all-VRAM, the token-critical path must
not execute tier splitting, CPU/NVMe bookkeeping, joins, or placement
decisions. Recorded rider from the same receipts: after the collapse,
R1's own shape (13.3 ms kernels inside a 49.4 ms step) says the M=1
executor ladder (BRANCH-3-HOST work: device-resident routing metadata,
op fusion/launch collapse) is queued directly behind it — and per the
prereg's native-code rule, a native boundary is licensed only if that
structural collapse still leaves the step multiples of its kernel
occupancy.

Answer to the owner's question: the B=1 deficit is (b) hybrid-engine
orchestration ≈ 25%, then (c) framework/host dispatch (the dominant
remainder — 3.7× kernels even in R1), with (a) heterogeneous residency
at only ~7% and (d) M=1 kernels a minor constant (~13 ms) that becomes
the floor only after the host work is gone.

## R1 mechanics (disclosed deviations)

- Module tensors materialized from the SAME arena file (byte-exact by
  construction, per-segment lengths asserted) because the streaming
  loader leaves meta stubs and the pipelined engine sources module
  tensors (#223).
- R1 ran `--chunk 1`: the engine's T>1 prefill falls back to the
  reference forward, which requires a second resident copy that does
  not fit in 32 GB; chunk=1 keeps every forward on the T=1 fast path.
  KV content is mathematically identical, enforced by the bitwise
  gate. Decode endpoints unaffected; prefill wall is not an endpoint.

## ERRATUM — vacuous token-identity gates in earlier cycles

The first B1 pass exposed that `generated_tokens` had been EMPTY in
every step_decomp rep since the field was introduced (t1 instruments):
the runner pops a finished request's tokens at release, and the rep
read the post-loop dict. Consequently the cross-arm token-identity
gates in the t1, t1b, and t5 cycles (and the first B1 pass) compared
empty records — G1 "pass" was vacuous. **No verdict flips**: t1, t1b
and t5 were REFUTED on performance grounds, so identity was never the
deciding bar; but RESULTS-t5's table row "G1 token identity —
bit-identical ×4 arms — PASS" overstated what was measured and is
corrected by this erratum (an erratum comment is posted on #219). The
capture is fixed in #223, and this cycle's identity results are the
first REAL ones on this instrument.
