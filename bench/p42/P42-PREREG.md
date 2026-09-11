# Pre-registration — lane P42: where the B=16 decode step actually goes

Registered 2026-09-11, before any rental. One box, one RTX 5090
(`vast:verified-secure`, the class P37 and P39 measured on), launched from the mini
through `adertha-agents/tools/pod-launch.sh`.

**Owner authorisation.** Jordan, working session 2026-09-10, approving five decisions,
the fifth of which was: *the B=16 levers before an MXFP4-specific plan.* That is the
top of the approval band and it does not waive this document or the STOP rules below.
Written down by the agent, not by Jordan.

## Why this lane exists, and what it is not

On the same box, same model (`Qwen/Qwen3-30B-A3B` @ `ad44e777…`), same prompts:

| | B=1 | B=16 |
|---|---|---|
| ours (P39 box 4 / box 1-3) | 4.820 ms/step | 12.390 ms/step |
| vLLM 0.28.0 `graph_r1` (P37) | 3.497 | 7.882 |
| vLLM 0.28.0 `fp8kv` (P37) | 3.323 | 7.251 |

So at B=16 we are **1.57×** behind `graph_r1` and **1.71×** behind `fp8kv`, and
closing it means finding about **4.5 ms** in a 12.39 ms step.

I have been carrying two candidate levers — a fused permute/unpermute worth roughly
1.5 ms and a small-M int4 attention GEMM worth roughly 1.7 ms. **Neither number is
backed by any receipt in this repository.** I could not find the census they came
from. So this lane measures rather than assumes, and the two figures are registered
below as predictions that can be wrong.

**This lane changes no kernel.** It produces an attribution and nothing else. Whatever
it finds, the optimisation work is a separate lane with its own pre-registration.

## Design

One box, four arms, a 2×2 of `{nf4, int4} × {B=1, B=16}`. Each runs
`step_decomp.py --replay-profile-out`, which times 8 graph replays *after* the timed
window, in KV slots `cap_tokens` already reserves, so the timed number stays
byte-identical to a no-flag run.

The 2×2 is the point. `int4_b16` alone gives a list of kernels; the other three say
which entries are **batch-scaling** (B=1 → B=16 at fixed path) and which are **the int4
path's own** (nf4 → int4 at fixed batch). A lever has to be one or the other.

The census at B>1 was unreachable until #555: the BV3 stage gated it on `--b1d-timed`,
a B=1 flag, so `--replay-profile-out` silently wrote nothing. This lane does not launch
until #555 is merged and `staged.sha256` re-pins the fixed `step_decomp.py`.

### The int4 arms are deliberately uncalibrated

Packing round-to-nearest instead of running GPTQ removes ~20 minutes and most of the
cost. The census measures which kernels run and for how long; gptq vs rtn changes the
weights those same kernels read, not the kernels themselves.

**The check that keeps this honest:** `int4_b16`'s `step_ms_clean` must land within
**5 %** of p39-box4's calibrated **12.390 ms**. Outside that, the census describes a
configuration we have not been quoting, and the lane says so instead of analysing it.

## Registered predictions, and what refutes them

1. **The two carried levers.** Fused permute/unpermute ≈ 1.5 ms/step and small-M int4
   attention GEMM ≈ 1.7 ms/step, at B=16. *Refuted by:* the corresponding kernels
   costing outside 1.0–2.0 ms each, or not appearing as separable rows at all. I expect
   at least one of these to be wrong, because neither has a source.
2. **The expert GEMV dominates.** `_gemv_int4_b32` is the single largest row at
   `int4_b16`. *Refuted by:* any other kernel family costing more.
3. **Batch scaling is superlinear somewhere.** At least one kernel family costs more
   than 16× its B=1 cost. *Refuted by:* every family scaling at or below 16×.

## The decision rule, registered in advance

After the census I will pursue **the largest attributable cost that is not the expert
GEMV itself**, whichever it turns out to be, and I will say plainly if my two carried
levers are not in the top three. Registering this is the point: without it, a census
with 40 rows lets anyone find support for the lever they already wanted.

A row only counts as a lever if it is **separable** — a kernel family that could be
fused, replaced or removed without changing the arithmetic the model performs.

## Accounting standard

Per-kernel cost is self-CUDA divided by the profiled replay count, so the unit is
milliseconds of one decode step. The census's own percentages are percentages of the
census, a different denominator, and are not quoted. `p42_reduce.py` prints the
**accounted fraction** against the timed step and labels its regime: under 90 % is a
hint rather than an attribution; over 100 % means kernels overlap across streams and
the rows are relative weights, not a budget.

## Budget and STOP rules

- **One box, ≤ $1.00, ≤ 1.5 h wallclock.** Lane ceiling $3, hard stop $5.
- **STOP-1** — `int4_b16` step time outside ±5 % of 12.390 ms: record the number,
  publish the census as describing a *different* configuration, draw no lever from it.
- **STOP-2** — an arm cannot finish 10 minutes before the launcher's deadline: skipped
  and recorded host-limited, never shortened.
- **STOP-3** — any arm's census missing: the run exits non-zero. A lane whose only
  output is a census does not get to report success without one.
- **STOP-4** — no second box on a disappointing result. If the census is
  uninformative, that is the finding.

## Receipts

Fetched to the private record under `receipts/experts4bit-qlora/<date>/p42/`:
`logs/census_<arm>.txt` ×4, `e4b_b{1,16}_<arm>.json` ×4, `summary.txt`, `forensics.txt`,
`versions.txt`, per-arm VRAM traces. `RESULTS-p42.md` is generated by `p42_reduce.py`
from those files and from nothing else.

---

## Amendment 1 (2026-09-11, after run `p42-census-2` was stopped)

**My design error, and the fix.**

The prereg argues that the int4 arms can skip GPTQ because "the census measures which
kernels run and for how long; gptq vs rtn changes the weights those same kernels read,
not the kernels." That reasoning is right and I applied it to the experts only. The run
script tied the attention flag to the same switch as the experts, so every int4 arm
asked for **calibrated** int4 attention — a Hessian pass over 192 projections, which is
a build cost, not a serve cost.

P39 never noticed because it paid that pass **once**, in its build arm, and every later
arm loaded the result from an artifact. This lane has no artifact, so each int4 arm was
paying it fresh.

Measured: `p42-census-2`'s `int4_b1` reached `INT4EXP enabled` in **21 s** — the RTN
expert pack is cheap, as predicted — and then sat in attention calibration for **40
minutes** without reaching a timed window, while the same host ran the two nf4 arms in
about four minutes each. I stopped the run through the script's own `TERM` trap so the
two finished censuses were fetched, and the box was torn down. rc=130, $0.4853.

**The fix.** The int4 arms take `E4B_SERVE_ATTN_INT4=1`, the uncalibrated path.
`enable_serve_attn_int4` and `enable_serve_attn_int4_calib` install the **same**
`Int4Linear` class from `int4_attn.py` and differ only in the packer (round-to-nearest
from the source bf16 vs GPTQ with a Hessian). The serve-time kernels — the whole subject
of a census — are identical. The P39 hook wired only the calibrated flag, so this lane
now carries its own hook with an uncalibrated branch.

**Cross-check, registered:** the uncalibrated swap must report **192 projections**, the
count P39's `ATTNINT4 calibrated:` lines record on this model. A different count means
the two paths do not cover the same modules and the census is not of the configuration
we quote. The run writes that count to `summary.txt` per arm.

**What this does not change.** The 12.390 ms target, STOP-1's ±5 % band, the decision
rule, and the accounting standard all stand. The two nf4 arms are re-run rather than
carried over from `p42-census-2`, so the 2×2 comes from one box.

**Budget.** `p42-census-2` spent $0.4853 of the lane's $3 ceiling. The re-run is estimated
at $0.60, for about $1.09 against that ceiling and the $5 hard stop.
