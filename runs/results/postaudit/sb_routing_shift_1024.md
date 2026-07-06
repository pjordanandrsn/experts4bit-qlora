# S-B — adapters steer routing (n=1024, base vs adapted, resident)

Lane S-B (`docs/SPECULATIVE_LANES_PLAN.md`), measured on the pinned n=1024 confirmation set:
each mode's seed-0 portability adapter evaluated on its train-precision base with routing
telemetry, compared per example against the matching base-mode ∅ telemetry (same eval set).

| adapter | mean J(base, adapted) | routing shift (1−J) | corr(shift, \|Δloss\|) | eval base → adapted |
|---|---|---|---|---|
| nf4 | 0.9418 | 0.0582 | +0.583 (Spearman) | 1.6188 → 1.2249 |
| int8 | 0.9447 | 0.0553 | +0.603 (Spearman) | 1.6022 → 1.2162 |

## Committed predictions (SPECULATIVE_LANES_PLAN.md S-B), graded

- **J(adapted, base) ∈ [0.85, 0.97] — HOLDS** (0.942 / 0.945).
- **shift correlates with per-example adaptation gain — HOLDS, strongly** (ρ 0.58 / 0.60):
  the examples the adapter helps most are the examples whose routing it moves most. This is
  the graduation condition — the mismatch mechanism gains its second term (an adapter trained
  on base A moves routing computed on A; a different base B routes differently), and the
  routing-pinned-serve experiment is promoted with a mechanism-complete story.
- **shift LARGER than int8-vs-bf16, SMALLER than nf4-vs-bf16 — FAILS (informatively).**
  Adapted shift 0.055–0.058 exceeds BOTH precision perturbations (int8-vs-bf16 0.0045,
  nf4-vs-bf16 0.0194 from the n=1024 factor probe). The adapter is the **dominant** routing
  mover on this stack — larger than any weight-precision change — not a mid-sized one. A
  cleaner result than the bracket predicted.

## Reading

Adaptation on a frozen-router MoE is not expert-content-only: ~5.6% of layer-expert routing
decisions move, and the movement tracks where the adapter earns its loss reduction. This is
the first base-vs-adapted routing-shift number for this stack (S-F's provenance-fingerprint
feasibility datum, recorded in passing). It does not by itself license a serving change —
that is the promoted routing-pinned experiment, gated as any new mechanism is.
