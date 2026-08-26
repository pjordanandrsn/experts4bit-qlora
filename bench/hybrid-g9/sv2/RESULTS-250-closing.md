# RESULTS — 250 tok/s single-stream: NOT closed.

## Speculation is refuted. Composition remains OPEN, pending the one
## lane nobody has measured.

Written 2026-08-26; **corrected the same day, before merge** — the
first draft of this document called 250 REFUTED-AS-COMPOSED and was
wrong. The error and its correction are recorded below rather than
edited away, because the error is the same one the document was
written to diagnose.

No new measurement: this adjudicates a registered pre-commitment
against receipts already committed by SV2, K7, K8 and K10.

## The rule being applied

PREREG-sv2 registered two disjoint routes and a closing condition:

> **250 itself is REFUTED-AS-COMPOSED only if BOTH routes fall
> short** — and the RESULTS must say so rather than stretching either
> side's estimates.

### Route 1 — speculation: CLOSED, twice, by measurement

S3 measured grouped speculative verify at 0.60–0.66× (a LOSS at every
K). SV2 re-measured it on the certified batched-graph stack with
matched acceptance: best cell **11.44 ms per accepted token against a
4.00 ms bar**, 2.9× short. F2, dot-pad and bitwise device grouping
moved verify by ~0.1 ms. This route is closed.

### Route 2 — composition: OPEN. Its dominant lane is unmeasured.

| lane | SV2 framed | status |
|---|---|---|
| **MoE GEMV round 2** — *tensor-core mapping beyond dot-pad's 15/16 M-row waste* | **1.5–1.8 ms** | **NEVER MEASURED** (see below) |
| fp8-COMPUTE attention | 0.15–0.20 ms | **0.217 ms measured** (K8 PASS, ships OFF) |
| router sort | inside "residue" | **0.000 ms** (K10 B1 REFUSED — 0.132 ms real, sets diverge) |
| router select | inside "residue" | ≤0.104 ms (K10 B2 registered, unbuilt) |
| attn-proj GEMV | 0.22 ms | unmeasured; already 1.14× its floor |
| fusion/norm residue | 0.7–0.8 ms | unmeasured, never registered |

## The correction: K7 did not measure SV2's MoE lane

SV2 registered that lane as **tensor-core mapping past dot-pad's
M-row waste**. K7's prereg scoped itself explicitly *out* of it:

> The treatment is **parallelism and pipelining on the SAME dequant
> math** […] Explicitly OUT of scope: fp8-MMA activations […] any
> dequant-chain change (the bf16-MMA rounding mechanism and absmax
> pre-fold stay IDENTICAL to the certified K6-B chain).

K7 tested a *different hypothesis about the same slice* — that the
kernel was occupancy-starved — and refuted it cleanly. What it
produced (0.241 ms) is a **config retune**, real and available, but
it is not the registered treatment and does not bound it. K7's own
RESULTS says so: *"the 3.8×-floor gap is still there and still real;
a different mechanism … may yet reach it."*

The first draft of this document nonetheless entered 0.241 ms as the
lane's measured value and declared the route refuted. That is
precisely the failure it goes on to name — **an unmeasured dominant
term leaves the verdict unmeasured** — committed while writing the
sentence that names it. Caught in review (Bugbot, e4b#282).

## What the verdict actually hinges on

Grant every *other* lane its best framed value, plus K7's retune:

```
K8 fp8 attention (measured)                 0.217 ms
K10-B2 at its registered PASS ceiling       0.104 ms
attn-proj at its framed value               0.220 ms
fusion residue at the HIGH end              0.800 ms
K7 config retune (real, not the lane)       0.241 ms
                                    total   1.582 ms
bar                                         2.480 ms
        the registered MoE lane must supply 0.898 ms
```

How much headroom is left depends on whether K7's retune survives a
round-2 kernel — and the retune came out of the SAME 2.469 ms slice,
so banking it *and* granting the lane the full pre-retune headroom
would spend that slice twice (Bugbot, e4b#282):

| reading | lane must supply | out of | share |
|---|---|---|---|
| **A** — the retune composes with a round-2 kernel | 0.898 ms | 1.579 ms remaining (2.228 slice − 0.649 floor) | **57%** |
| **B** — a round-2 kernel brings its own config and supersedes the retune | 1.139 ms | 1.820 ms | **63%** |

So **250-as-composed requires the tensor-core mapping to capture
57–63% of the headroom that is actually left** — not the 49% this
document claimed on its first correction, which double-counted the
retune. Reading B is the more likely one: a new kernel picks its own
BLOCK_N/warps/stages, so K7's config win on the *current* kernel
probably does not carry.

Not demonstrated. Not excluded. That is the whole finding.

## Verdict

**250-as-composed is NOT refuted, and is NOT on track.** Its status
is OPEN-PENDING-ITS-DOMINANT-TERM — the label this document argued
SV2 should have used, and which applies to this document too until
the registered MoE treatment is built and measured.

Closing it in either direction requires exactly one thing: a
registered cycle that implements tensor-core mapping past the M-row
waste and measures it. A result below ~0.90 ms (reading A) or ~1.14 ms (reading B) closes
250 by arithmetic. A result at or above it keeps the target live and
makes the remaining unmeasured lanes worth registering.

## What is certified regardless

| configuration | ms/step | tok/s |
|---|---|---|
| certified default | 7.35 class | ~136 |
| `GNF4_GEMV_DOTPAD=1` | 6.476 | 154.4 |
| **+ `GNF4_ATTN_COMPUTE=fp8`** | **6.281** | **159.2** |

(The default row is under re-certification by PREREG-m2; the two knob
rows are same-box measurements and do not depend on it.)

## The lesson, stated against this document

A composition frame's verdict is only as measured as its largest
term. SV2 left one lane at 1.5–1.8 ms unmeasured and called the route
NOT-REFUTED; this document left the same lane unmeasured and called
it REFUTED. Both directions are the same error. The lane has now been
named precisely enough — *tensor-core mapping past the M-row waste,
needing 57–63% of the headroom that remains* — that the next cycle can
settle it instead of estimating it.
