# PREREG — T1b: reduce-overhead with step marking (SPEC-400, gate 1 retry)

Registered before any measurement. T1's compile route REFUTED in
default mode (+5.5 ms: ~100 compiled-region entries/step of guard
overhead beat the fusion gains — [RESULTS-t1-launchpath.md](RESULTS-t1-launchpath.md))
and its cudagraph mode crashed on replay reuse across the residual
stream. T1b applies the crash message's own documented remedy:
`torch.compiler.cudagraph_mark_step_begin()` at the top of every model
invocation (prefill and decode), active only when `--compile-mode`
contains reduce-overhead. Cudagraphs are the ONE compile mode that
eliminates the per-entry cost T1 measured, so this is the compile
route's last registered attempt.

## Bars (unchanged from T1 in structure; baseline = this box's own
eager arms)

A/B/A: eager, compiled (`--compile-layers`, reduce-overhead), eager —
the kvappend operating point, `--kv-batched` on everywhere.

* **B0 (void)**: token-identical greedy continuations across all arms;
  equal uniques. A cudagraph crash in the compiled arm is a REFUTATION
  of T1b (no second fallback — default mode was already scored).
* **B1**: compiled other-submission ≤ **25 ms/step**.
* **B2**: compiled step ≤ **85 ms/step**, scoreable at 3× the A/A
  spread.

REFUTED (crash, divergence, or either bar) closes the compile route
for good; the ladder proceeds to T2/T3 (device kernels) and T1c
(manual op-count reduction) stays as the launch-path fallback. One
box, one scored A/B/A. ≈ $0.80.
