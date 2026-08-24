# PREREG — T2/T3 attribution probe (SPEC-400/425, measurement first)

Registered before the measurement. Owner directive raises the target:
**425+ tok/s at B=16 = a 37.6 ms step**. The remaining budget after
the kvappend gate is dominated by two DEVICE buckets — attention
kernels (~28 ms/step measured by CUDA events) and GPU expert kernels
(~30 ms) — each ~10× above its bandwidth floor. Per the hostbill
precedent (whose cProfile pass overturned the "python" hypothesis and
named the real disease), neither kernel fix is registered until the
device time is ATTRIBUTED: a 28 ms bucket could be one slow kernel, a
storm of small ones, or aux ops (gathers, dequants, indexing) around a
fast core — three different fixes.

## The instrument (this PR)

`step_decomp --torch-profile-out`: ~12 decode steps under
torch.profiler (CPU+CUDA, per-step schedule with warm-up skip), CUDA
kernel table by total device time with launch counts, dumped verbatim.

## Go/no-go (frozen)

* **T2 (attention kernel) opens** iff the table attributes ≥ 60% of
  the attention-bucket device time to ≤ 3 kernels — a concentrated
  target. If instead the bucket is aux-op spray (gathers/copies/
  dequant helpers), the fix is a fusion line and registers as such.
* **T3 (expert path) opens** the same way over the expert-bucket
  kernels; a launch-count ≫ layers signature (many small grouped
  launches) routes T3 to launch batching, a slow-single-kernel
  signature routes it to kernel config/occupancy work.
* Probe receipts committed regardless; each opened line freezes its
  bar from its own attribution (the T1 pattern: bar = half the
  attributed share).

## Cost

Rides the T1b rental after its arms (marginal ≈ $0.10); one profiled
run at the operating point.
