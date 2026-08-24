# PREREG — G9 host-bill: the attention/python step tax, measurement first

Registered before any measurement. The productionization receipts moved
the bottleneck: at B=16 the ~194 ms decode step is ~119 ms
attention-host + ~80 ms other submission/python against only ~12–14 ms
of CPU expert decode — so the slot controller's certified −20.8% on the
expert bill buys just +1.3% end-to-end. This is the HANDOFF's open G9
engine item ("non-expert step time: attention kernels + python
dispatch"), and per the routing-overlap precedent it opens
MEASUREMENT-FIRST: no fix is registered until the bill is attributed.

## Phase 1 (this registration): attribution

On the same box as the G8-close run, `step_decomp.py` runs at the
serving operating point (B=16, chunk 512, pool 32, gen 128) twice:

1. **Bucket pass** (existing instrument): step / forward-submission /
   attention-host / dram / other-submission / drain / scheduler-python
   medians, plus device-side attention and GPU-expert occupancy.
2. **cProfile pass** (`--cprofile-out`, added by this PR): the same
   loop under cProfile, top-60 functions by cumulative time. The
   profiler inflates the wall (reported, not compared against pass 1);
   its job is per-function ATTRIBUTION of the python bill, which the
   bucket pass cannot see below the shim level.

**Go / no-go (frozen):** phase 2 (a targeted fix with bars) is
registered only if the two passes agree that ≥ 50% of the decode step
is host-side python outside CUDA kernels AND the top three attributed
functions cover ≥ 40% of that python bill — i.e., the tax is real and
CONCENTRATED enough for a targeted fix. Diffuse overhead (top-3 < 40%)
closes the line in favor of interpreter-level remedies (CUDA graphs /
compile), recorded as out of this program's scope.

## Phase 2 (promised shape, registered here, bars set from phase-1 receipts)

One fix targeting the top attributed component, with: (a) outputs
bit-identical on a fixed greedy continuation (the fix is host-side
overhead only — any numerical change voids); (b) attention-host +
other-submission reduced by a bar set as HALF the component's
attributed share (from phase-1 receipts, disclosed before the fix box
runs); (c) A/B/A wall with the discriminability rule. Phase 2 is its
own PR and box run.

## Hard stop

Phase 1 rides the G8-close rental (marginal cost ≈ $0.15); one bucket
pass + one profile pass, receipts committed regardless of go/no-go.
