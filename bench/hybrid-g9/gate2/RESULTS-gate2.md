# Gate attempt 2: work-stealing redeems the fused kernel; fixed=55 confirmed; absolutes await the reference host

EPYC 9554 64c (185 GB) + RTX PRO 6000 Blackwell, $0.934/hr, capped
harness, tee'd stages, all exactness on-box (38/38 x2, 9/9). Stack under
test: work-stealing (gnf4#110) x fixed-term sweep x thin routing x
fused re-test x warm G9. Box destroyed; SSH-refused + API-null verified.

## Same-box arm comparison (the valid signal on this host class)

| arm (all thin=4, capped, 32w) | B=8 dram / ratio | B=16 dram / ratio |
|---|---|---|
| fx55, two-call | 43.1 ms / 0.540 | 41.7 ms / 0.570 |
| fx100, two-call | 49.9 / 0.456 | 41.0 / 0.549 |
| fx150, two-call | 32.7 / 0.674 | 47.1 / 0.481 |
| **fx55, FUSED** | **33.6 / 0.695** | **40.2 / 0.578** |

- **Work-stealing redeems the fused kernel in-executor: best arm at
  both batch sizes, +22% over two-call at B=8** (33.6 vs 43.1) — the
  dev-box prediction (coarse row-chunk items steal-balance) transfers.
  One call per layer is again the right shape now that the partition
  tail is gone.
- **Deeper concentration does not pay**: fx100 is flat-to-worse than
  fx55; fx150's split verdict (0.674 @8 / 0.481 @16) tracks this host
  class's known 2x wobble, not a law. fixed=55 stays the constant.
- G9 here: 36.7 tok/s, TTFT p50 6.04 s (warm fix holding).

## What binds and what doesn't

This is the wobbly host class from gate-1 with a different GPU (PRO
6000; its fused-GPU path read 43-53 GB/s vs the 5090's 55-63), so the
ABSOLUTE ratios do not adjudicate the 0.80 bar. The binding facts:
fused+stealing is the best executor shape, fixed=55 is the constant,
and on the reference TR host — where CTRL measured 26.6 ms / 0.675
PRE-stealing — the stack's same-box gains (stealing's in-call win plus
fused's +22%) project the bar within reach. The final adjudication is
one run of THIS stack on the quiet TR when the offer recurs.

Consequence deferred, deliberately: `fused_ffn` stays default-False
until the reference host confirms the flip — one wobbly-class
measurement should not toggle a default twice in one day.

Receipts: `g8_fx{55,100,150}.json`, `g8_fused.json`, `g9_gate.json`,
`calib.json`, `rows_curve.json`.

## Post-hoc: a deadline-estimator defect, and why it does not reach this result

Bugbot (e4b#179) found that `_cold_to_cpu_deadline` charged DRAM work to
the CPU side unless `_gpu_only`/`dram_thin` was set, while `_cold_contrib`
could *also* route DRAM to the GPU via the `offload_rows` threshold — and
that GPU load was charged to neither side. An estimator in that state sees
a busy CPU beside an idle GPU that is in fact doing the DRAM work.

**It cannot explain this gate's MISS.** The receipt (`g9_gate.json`)
records `offload_rows = null`, and neither `dram_thin` nor `_gpu_only`
appears in any arm's config. With no threshold set the DRAM rows went to
the CPU on every step, which is exactly what the estimator assumed, so its
inputs were correct throughout. The 0 destination flips out of 1975
decisions are a real property of the measured regime, not an artifact of
mis-attributed backlog.

Stated because the alternative — fixing it silently — would leave a reader
who finds the fix later unable to tell whether the published numbers were
taken before or after, and therefore unable to trust them. The defect is
fixed on the branch that carries this paragraph; the gate is NOT re-run,
because re-running a configuration the bug never touched would produce a
receipt that differs only by noise.
