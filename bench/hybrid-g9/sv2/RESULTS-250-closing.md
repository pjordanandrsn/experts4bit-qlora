# RESULTS — 250 tok/s single-stream: REFUTED-AS-COMPOSED.

## Adjudicating SV2's pre-commitment against four cycles of receipts.

Written 2026-08-26. No new measurement: this closes a registered
pre-commitment using receipts already committed by SV2, K7, K8 and
K10.

## The rule being applied

PREREG-sv2 registered two disjoint routes and a closing condition,
before any of them was measured:

> **250 itself is REFUTED-AS-COMPOSED only if BOTH routes fall
> short** — and the RESULTS must say so rather than stretching either
> side's estimates.

### Route 1 — speculation: refuted twice

S3 measured grouped speculative verify at 0.60–0.66× (a LOSS at every
K). SV2 re-measured it on the certified batched-graph stack with
matched acceptance: best cell **11.44 ms per accepted token against a
4.00 ms bar**, 2.9× short. F2, dot-pad and bitwise device grouping
moved verify by ~0.1 ms. Closed by direct measurement, twice.

### Route 2 — composition: now falls short too

SV2 left this route open on **estimates**. Three of its four lanes
have since been measured:

| lane | SV2 framed | measured | source |
|---|---|---|---|
| MoE GEMV round 2 | 1.5–1.8 ms | **0.241 ms** | K7 REFUTED — split-K flat; the 9.8% that appeared was a config retune |
| fp8-COMPUTE attention | 0.15–0.20 ms | **0.217 ms** | K8 PASS — shipped OFF, +0.0092 ppl |
| router sort | (inside "residue") | **0.000 ms** | K10 B1 REFUSED — 0.132 ms real, but the expert SETS diverge |
| router select | (inside "residue") | ≤0.104 ms | K10 B2 registered, unbuilt — ceiling is 60% of a 0.174 ms slice |
| attn-proj GEMV | 0.22 ms | unmeasured | already 1.14× its streaming floor; little to take |
| fusion/norm residue | 0.7–0.8 ms | unmeasured | never registered |

**The decisive arithmetic.** Grant every unmeasured lane its **best**
framed value *and* grant B2 a full PASS it has not earned:

```
measured (K7 0.241 + K8 0.217 + K10-B1 0.000)   = 0.458 ms
+ B2 at its registered PASS ceiling              = 0.104 ms
+ attn-proj at its framed value                  = 0.220 ms
+ fusion residue at the HIGH end of its estimate = 0.800 ms
                                          total  = 1.582 ms
bar                                              = 2.480 ms
                                        SHORTFALL  0.898 ms
```

**36% of the bar is unaccounted for under assumptions chosen to
favour the target.** Both routes fall short, so per SV2's own rule,
**250-as-composed is REFUTED.**

## What is true instead

| configuration | ms/step | tok/s |
|---|---|---|
| certified default | 7.35 class | ~136 |
| `GNF4_GEMV_DOTPAD=1` | 6.476 | 154.4 |
| **+ `GNF4_ATTN_COMPUTE=fp8`** | **6.281** | **159.2** |

(The default row's absolute value is under re-certification by
PREREG-m2; the two knob rows are same-box measurements and do not
depend on it.)

Reaching 250 needs 4.00 ms/step — another **2.28 ms** off the current
best. The largest single opportunity on the device remains the MoE
GEMV at **3.8× its streaming floor**, and K7 refuted one hypothesis
about why (occupancy) without touching the gap itself.

## Why the frame was wrong, stated plainly

The composition route survived SV2 on the strength of one lane
estimated at 1.5–1.8 ms that measured 0.241. Everything else was
close: K8 came in slightly ABOVE its estimate. The frame was not
uniformly optimistic — **its dominant term was an estimate, and it
was 7× off.**

The lesson is about frame construction, not about optimism: a
composition frame whose largest term is unmeasured is a frame whose
verdict is unmeasured. SV2 labelled its estimates honestly and
pre-committed to recomputing on measurement, which is why this
document can be written at all — but the route should have been
called OPEN-PENDING-ITS-DOMINANT-TERM rather than NOT-REFUTED.

## What would reopen it

Not a better version of anything in the table above. Reopening 250
requires a mechanism **outside** the registered frame that addresses
the MoE GEMV's floor gap — a different tiling, a weight-stationary or
lower-precision compute path, or a dequant-chain change carrying its
own numerics frame. Any such candidate is a new registration with its
own bars, and this document is not evidence for or against it.

## Status of the campaign's single-stream program

- **Closed:** speculation (twice, by measurement); 250-as-composed
  (here, by its own registered rule).
- **Open and certified:** 159.2 tok/s, reproducible, with receipts.
- **Open and registered:** K10-B2 (fused exact top-k, ≤0.104 ms);
  M2 (anchor re-certification, may restate the default row).
- **Open and unregistered:** the fusion/norm residue (0.7–0.8 ms
  estimated, never measured) and the MoE floor gap itself.
