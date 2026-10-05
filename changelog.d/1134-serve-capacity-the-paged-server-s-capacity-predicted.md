### `serve_capacity`: the paged server's capacity, predicted from its own step costs (new module, no default changes)

- **Why.** `serve_recipe` prices what `serve_paged` allocates. Nothing predicted what it delivers, so a planner could
  not answer whether a workload is servable at an SLO on this box, or which `max_seqs` it needs.
- **What.** `experts4bit_qlora.serve_capacity`, a model of `ContinuousScheduler` as `serve_paged` drives it:
  - `StepCosts` holds the step costs. `StepCosts.from_step_trace` reads them from a server's own
    `E4B_PAGED_STEP_TRACE` and refuses rather than invent a missing one;
  - `Workload` draws plans exactly as `bench/sc2/sc2_driver.plan` does;
  - `simulate` returns per-request TTFT / TPOT and SLO attainment;
  - `ceiling` gives SC2's ceiling rule under the model.
- **Checked against receipts.** From SC2b's two ON servers' own step costs it reproduces their measured attainment at
  1 / 2 / 4 / 8 req/s within 0.10. The worst gap is +0.093 at 4 req/s, where the model is optimistic, so its ceiling
  near the knee is an upper bound.
