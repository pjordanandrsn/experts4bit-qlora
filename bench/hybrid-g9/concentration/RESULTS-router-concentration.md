# The router-concentration solver experiment: closed by proof — the solver already minimizes uniques

The surviving P4-line lever from gnf4
(`bench/cold-engine/RESULTS-p4-cellmodel.md`: call cost ≈ fixed +
58 µs × uniques + 2.4 µs × rows, so "concentrate rows per unique") run
down against the solver's actual objective. Offline; no box; the
verdict is structural.

## 1. The constants cross-validate — two instruments, one physics

`solve_placement` books CPU cost as `w × cpu_us_fixed + rows ×
cpu_us_per_row` with **w = P(touched) — per-unique-appearance, exactly
the cell model's unit**. The campaign's in-situ constants
(`cpu_us_fixed = 55, cpu_us_per_row = 2`, fixbox receipts) and gnf4's
independent cell-model fit on the single-expert T-sweep
(**57.8 µs/expert-cell, 2.37 µs/row** serving marginal) agree to ~5%.
Two different harnesses, two different hosts, one cost law — both
instruments are strengthened.

## 2. The concentration re-ranking cannot exist — monotonicity proof

Expected DRAM uniques per step = Σ_{e ∈ DRAM} P_e(touched). Minimizing
it over a VRAM budget of S slots means placing the S highest-P experts
in VRAM. `P = 1 − (1 − p)^B` is strictly monotone in routed mass p, so
**top-mass ordering — what the greedy already does — IS top-P ordering**:
the frequency-ranked placement and the mass-ranked placement are the
same placement. The imagined experiment ("rank by call frequency instead
of row mass") proposes a distinction without a difference. And
independently, B=16 VRAM is capacity-clamped (4045 slots,
`b16close/RESULTS-b16.md`): no objective change moves any expert.

## 3. What the decode bill is actually worth, and what could still claim it

At B=16: 188 DRAM uniques/step × 58 µs ≈ **10.9 ms of the 22.9 ms CPU
wall is per-unique decode** — the bill is real and large. The remaining
mechanisms that could reduce uniques-per-call, priced honestly:

* **Batch composition** (schedule requests with similar routing into the
  same step): the only lever that changes the DRAW rather than the
  placement. New-architecture territory; would need its own registration
  with a routing-similarity measurement first (expected overlap of top-k
  sets across requests — measurable offline from any per-request routing
  capture before any engine work).
* **More VRAM slots** (FP8/param-quant workstreams already on the open
  list): every slot removes its expert's P(touched) × 58 µs/step.
  At the tail's P ≈ 0.3–0.6, each additional slot is worth ~17–35
  µs/step — a concrete slot-value curve the FP8 certification work can
  now cite.
* **The 2.4 µs/row multi-expert interaction term** (gnf4's bounded
  residual): orthogonal to uniques; its own registration or none.

## Disposition

The experiment closes REFUTED-BY-STRUCTURE: the proposed objective
change is a no-op by monotonicity and doubly a no-op by capacity. The
positive yields: the constants cross-validation (§1), the slot-value
curve (§3), and the batch-composition lever now stated with its offline
pre-measurement. Per the standing discipline, that pre-measurement —
routing-overlap statistics across requests — is the next
registered-or-nothing step on this line.
