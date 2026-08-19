# Gate attempt 1: the torch-thread-thrash law, a thin-route win, and a box that wobbles

EPYC 9554 64c (62 GB, 60.2 GiB cgroup, 61.4-core quota) + RTX 5090,
$0.401/hr. Both levers merged and deployed (gnf4#109 subset engagement
+ race fix, e4b#159 thin-layer routing). Box destroyed; SSH-refused +
API-null verified.

## The finding that matters beyond this box: torch intraop threads
fight the pool, scaled by the box's PHYSICAL core count

The executor read 0.7 GB/s from stacks that kernel-only reads at
33 GB/s on the same box (g8_diag receipts). Cause: torch defaults its
intraop pool to PHYSICAL cores — 64 here — and those unpinned threads
trample the 32 pinned pool workers during every forward. Cap sweep
(CTRL arm, B=8):

| torch threads | dram ms | ratio |
|---|---|---|
| 64 (default) | 630 | 0.038 |
| 24 | 145 | 0.160 |
| 16 | 126 | 0.194 |
| 8 | **52.6** | **0.458** |

12x from the cap alone. This variable was SILENTLY set by every prior
box's core count: the TR PRO (32 physical) defaulted to 32+32 = exactly
its 64 hardware threads (mild), and its w48/w64 "oversubscription"
collapse was the same law from the pool side. Standing instrument rule:
**harnesses must pin torch intraop (~8 threads) whenever the pool is
engaged** — now baked into the bench scripts.

Co-tenant theory of the first-pass collapse is RETRACTED (load 1.15
reproduced the sickness; the earlier high loadavg was our own setup).

## Thin-layer routing: mechanism proven, same-batch win at B=16

thin_layers=32 engaged at threshold 4 (the concentrated placement's
early layers), 49-70 calls/measurement rerouted, moved work visible on
the GPU-side clock. Same-process comparison at cap-8, B=16:
CTRL 105.5 ms vs THIN 88.1 ms — **16% DRAM-wall reduction** (ratio
0.224 -> 0.279). B=8 pairs were split across the box's wobble (52.6 vs
104 ms on IDENTICAL CTRL config minutes apart) and are not quotable.

## G9 with every instrument fix at once

Warm-both-shapes + cap-8 + thin routing: **33.5 tok/s, TTFT p50
7.05 s on a FIRST run** — the warm fix kills the 17 s compile
contamination as designed. Aggregate is below the TR's 44.0; this box's
GPU side read low all night (vram 42-49 GB/s vs TR 55-63) and the wall
wobbles 2x, so cross-box G9 deltas are not attributable.

Also fixed here: the gate pipeline's `| grep` swallowed a G9 crash
(stale `prefill_threads` kwarg from the reverted #158 patch) while
DRIVE_ALL_DONE printed — stage outputs now tee to files.

## Verdict

This box class (cheap, quota'd, 2x wobble even capped) cannot
adjudicate the gate. The clean adjudication is one run on the quiet
TR-class host with the capped harness: expected CTRL ~41 ms baseline,
minus the thin-route ~16%, ratio target ~0.55+ — with the remaining
distance to 0.80 owned by per-call floor structure now that the thrash
variable is controlled.

Receipts: cap sweep (`g8_CTRL.json` 64t, `g8_cap24.json`,
`g8_CTRL_capped.json` 16t, `g8_cap8.json`), quiet-window pair
(`g8_*_quiet.json`), cap-8 pairs (`g8_THIN_cap8.json`,
`g8_CTRL16_cap8.json`), `g9_gate.json`, `calib.json`,
`rows_curve.json`, kernel-only decomposition in `g8_diag.json` on-box
(not pulled — value quoted above from the live probe).
