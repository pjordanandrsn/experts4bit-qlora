# RESULTS — SV1: current-stack decode census + dot-pad×F2 composition

Adjudicated 2026-08-26. Box: instance 48709950 (post-F2 anchor
7.25 ms). Receipts in `receipts-sv1/`.

## Composition cert: PARTIAL (K6-B frame) — knob stays opt-in

```
off_a/off_b: 7.251 / 7.249 ms   (A/A spread 0.002 ms)
dotpad_on:   6.476 ms           (ratio 0.893; 127 tokens identical)
```

`GNF4_GEMV_DOTPAD=1` composes cleanly with the F2 defaults:
**154.4 tok/s single-stream** on this box (138 default). Ratio 0.893
lands in the K6-B PARTIAL band (≤0.85 PASS / ≤0.95 PARTIAL), same
tier as K6-B's own cert — the knob remains opt-in, now with a
composed receipt.

## Census: budget REFUSED coverage; busy floor NOT DELIVERED

The profiled census ran, but `step_budget.py` refused coverage
(recorded in `sv1_progress`), and the raw kernel table was then LOST
in teardown: the pull returned an empty file, the `[ -s ] || rm`
guard dropped it silently, and the box was destroyed on a file COUNT
rather than a manifest diff. Process failure recorded (memory +
teardown scripts now refuse destroy on a manifest mismatch).

**Consequence for the 275 target:** the pre-committed decision number
(busy floor vs 3.64 ms) is still unmeasured. It re-runs as a rider on
the BV3b box. The composition cert stands regardless: the certified
single-stream ladder is 138 default / 154.4 with the knob, and the
gap to 275 is 2.8 ms — undecidable until the floor lands.

## Addendum (2026-08-26): the busy floor — 275 REFUTED-AS-COMPOSED

<!-- BASIS CORRECTION (2026-08-26, appended after this addendum was
written; the addendum text below is left intact so the record of what
was claimed survives). The VERDICT stands: >275 is refuted-as-composed.
The two figures it argues from — 8.43 ms Self-CUDA and the 7.25 ms
graphed wall — both describe paths the certified ladder has since
superseded, and both sit ABOVE the certified opt-in's own 6.476 ms
wall. Device work cannot exceed wall-clock on a single stream, so the
correct bound for the certified path is 6.476 ms. See "Basis
correction" at the end of this file. -->

The re-run on box 48728047 delivered the lost number:
**Self-CUDA 8.43 ms/step** (101.169 ms over the 12-step active
window, `bv3b/receipts-bv3b/bv3b_kernels.txt.gz`; eager-path basis,
disclosed — the graphed 7.25 ms wall bounds device work in the same
class). Both readings sit far above the 3.64 ms line 275 requires:
**per the pre-commitment, >275 tok/s single-stream is
REFUTED-AS-COMPOSED for this stack.** Single-stream decode is
compute-bound; further gains there are kernel-compute work
(dot-pad-class), not orchestration. The certified single-stream
ladder stands at ~138 default / 154.4 with the knob; the throughput
lane is the batched graph loop (RESULTS-bv3b: 419 aggregate tok/s).


## Basis correction (2026-08-26): which number bounds the certified path

Appended after the addendum above, which is left unedited. **The verdict is
unchanged — `>275 tok/s` remains REFUTED-AS-COMPOSED.** What is corrected is the
number the refutation is argued from, and therefore the size of the margin.

### The inconsistency

Three figures have been used as bounds on single-stream device work:

| basis | box | anchor | ms/step | implied tok/s |
|---|---|---|---|---|
| Self-CUDA sum, **eager** path | 48728047 | 7.27 ms | 8.430 | 118.6 |
| graphed **default** wall | 48709950 | 7.25 ms | 7.250 | 137.9 |
| **certified opt-in wall** (dot-pad × F2) | 48709950 | 7.25 ms | **6.476** | **154.4** |

**Box provenance, stated because this correction would otherwise commit the error
it names.** The 8.43 census and the certified ladder are from *different rentals*:
8.43 on 48728047, 6.476 on 48709950. Their anchors agree to **0.3%** (7.27 vs
7.25 ms), so they are the same class under this project's own anchor discipline —
tighter than the 2.2% SV2's box was accepted at.

The argument does not actually rest on that cross-box step. It rests on a
**within-run** inequality: on the certified run itself, on 48709950, the wall was
6.476 ms, and device work on a single stream cannot exceed the wall-clock of the
run it occurred in. So device work on the certified path is ≤ 6.476 ms no matter
what any other box measured. The cross-box anchor agreement is corroboration, not
load-bearing.

With that said: the first two figures are *slower* than the third. Device work cannot exceed wall-clock on
a single stream, so once 154.4 tok/s is certified, the upper bound on device work
for that path is **6.476 ms** — not 8.43, and not 7.25. The 8.43 census bounds the
**eager** path (its basis is correctly disclosed above); the certified ladder no
longer runs it.

This is [[finding_gpu_busy_is_not_occupancy]] one level up: a summed device-time
statistic read as a bound on a configuration it did not measure.

### What changes

| target | requires | gap vs 8.43 | gap vs **6.476** |
|---|---|---|---|
| 275 tok/s | 3.636 ms | 4.794 ms — 2.32× | **2.840 ms — 1.78×** |
| 250 tok/s | 4.000 ms | 4.430 ms — 2.11× | **2.476 ms — 1.62×** |

1.78× of device-work reduction is not available from orchestration, so the
addendum's conclusion — *single-stream decode is compute-bound; further gains are
kernel-compute work (dot-pad-class), not orchestration* — **is correct and
unaffected.**

Note this file already stated the right frame three paragraphs above the addendum:
*"the gap to 275 is 2.8 ms"* = 6.476 − 3.636. The addendum's headline and this
body figure disagreed; the body was right.

### Do not re-derive the 250 verdict from 8.43

Against the certified wall the 250 frame needs **1.62×**, and SV2's composition
table already sums its lanes to ~3.0–3.3 ms. Consistent with `RESULTS-250-closing.md`
(#283) keeping that route OPEN. The dominant lane — MoE GEMV round 2, tensor-core
mapping past dot-pad's 15/16 M-row waste — remains **NEVER MEASURED**, so 250 is
neither reached nor refuted.

### External calibration

llama.cpp on RTX 5090, `qwen3moe 30B.A3B Q4_K - Medium`, ngl 99, fa 1, tg128:
**352.06 ± 1.76 t/s = 2.840 ms/token** (ggml-org/llama.cpp discussion #17621,
2025-11-30). With fusion and graph-opt disabled: 246.96 ± 0.31 t/s = 4.049 ms/token.

A comparable 4-bit engine on the same GPU and model already runs below both the
3.636 ms that 275 requires and the 4.000 ms that 250 requires. Neither target is
near a hardware floor — which is the same conclusion the addendum reached, now with
an external witness. *(Coincidence worth not tripping over: the 275 gap vs certified
is also 2.840 ms. Unrelated quantities.)*
