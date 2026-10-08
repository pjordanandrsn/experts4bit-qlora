### Router epilogue: the semantic probe ignores near-tied rows, and its real-router tests initialise their weights (fix; #1372 follow-up)

- **What changed.**
  - `router_epilogue._probe_matches` compares the routing (expert set and weights) only on decisive probe rows: those
    whose k-th and (k+1)-th selection logits are more than two ulps of the logits' dtype apart, relative to the k-th.
    It compares them per expert rather than per slot.
  - A router with non-finite logits on the probe, or with no decisive row, is refused.
- **Why.** `test_real_routers_keep_their_upstream_weight_dtype[mixtral]` failed intermittently in CI. transformers'
  `MixtralTopKRouter` allocates its weight with `torch.empty`, so the test's seed never reached it, and the probe then
  saw whatever memory held. `Qwen3MoeTopKRouter` allocates zeros, which ties every expert on every row.
  - A near tie at the k boundary can select either expert under the same function rounded differently, so it says
    nothing about which function a router computes.
  - An explicit `E4B_FUSE_ROUTER_EPI=1` could therefore have raised at random on such a row.
- **Who is affected.** Nobody serving: real routers on real weights read the same. The probe now licenses a router
  whose only disagreement with the reference is on near-tied rows.
- **Tests (`tests/test_router_epilogue.py`):**
  - the real-router tests initialise the weight from a seeded generator;
  - `_decisive_rows` on exact, sub-ulp and decisive rows;
  - a router that breaks an exact probe-row tie the other way from the reference is licensed on its decisive rows (and
    refused when every row counts);
  - a router whose experts all tie is refused, as is one with non-finite logits.
