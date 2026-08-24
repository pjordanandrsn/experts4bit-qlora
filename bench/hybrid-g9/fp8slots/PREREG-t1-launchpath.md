# PREREG — T1: the launch-path kill (SPEC-400, gate 1)

Registered before any measurement. Baseline frozen from
[receipts-kvappend](receipts-kvappend/): step 131.3 ms,
other-submission 76.4, attention_host 40.2, dram 14.0 (batched-append
arm; that switch is ON for every arm here).

## The mechanism

other-submission is the eager per-op launch path of the non-MoE,
non-attention layer body: RMSNorms, qkv/o projections, the router
linear+softmax+topk, residual adds — 48 layers × dozens of small ops,
each paying python + dispatcher before its (tiny) kernel. The cure is
tracing: `torch.compile(mode="reduce-overhead")` applied to the layer
body with the paged-attention call and the hybrid MoE forward left
OUTSIDE the compiled regions (both are dynamic/host-bound and must
graph-break cleanly).

## The increment

* A compile harness flag (`step_decomp --compile-layers`) that wraps
  each decoder layer's attention-adjacent dense body via
  torch.compile on the module forwards it owns (norms + projections +
  router), leaving `paged_attention_forward` and the tier forward as
  designed graph breaks. No engine-file changes: the wrapper is
  driver-side patching, receipts-only, so a REFUTED arm reverts
  nothing.
* Warm-up: the first FOUR decode steps are dropped from every timed
  median when compiling (compile lands across the first prefill and
  decode shapes, and reduce-overhead's cudagraph warm-up can spill into
  the second and third; the warm-up-inventory law). The receipt records
  the count actually dropped; device-kernel averages use the full step
  count, since device work is shape-stable through a host-side compile.

## Protocol and bars (one box, A/B/A: eager, compiled, eager)

* **B0 (void)**: the same 64-token greedy continuation, eager vs
  compiled, must be TOKEN-IDENTICAL. Any divergence refutes the arm
  regardless of speed (max-abs logit delta reported for the record).
  Uniques totals equal across arms (routing unchanged) — mismatch
  voids.
* **B1**: compiled-arm other-submission ≤ **25 ms/step** (from 76.4).
* **B2**: compiled-arm step ≤ **85 ms/step** (≥ 188 tok/s), scoreable
  iff the delta exceeds 3× the A/A step spread.
* Reported, unscored: attention_host under compile (should be ~flat),
  compile time, graph-break count, and the residual bucket table that
  re-freezes the ladder baseline.

REFUTED (any bar, incl. a token divergence) closes T1's compile route;
the fallback route (manual op-count reduction in the layer body) would
be its own registration. One box, one scored A/B/A. ≈ $0.90.
