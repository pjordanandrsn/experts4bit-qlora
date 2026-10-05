### `/health`'s `prefill_routes` gains `seen`: the routes the forward ran, not the environment's resolution (serving)

- **What was wrong.** `prefill_routes` reports the environment's resolution: `int4_prefill`, `int4_prefill_above_256_rows`
  and `prefill_attn`. Those name what an int4-b32 store and a layer without sinks or a window would take. On gpt-oss-20b
  (SC2g's `sc2g-prove-2`) they read `k19` / `k19` / `flash`, and no call took any of them:
  - the MXFP4 store's rows up to 256 take K21, and rows above take the kept NF4 stacks' M-tile GEMM;
  - every gpt-oss layer has sinks, so it keeps the explicit mask.

  Box G's engagement check asserted those names, so it passed without testing the route.
- **`prefill_routes.seen`.**
  - `moe` counts each expert-GEMM call's route and row class (`hot_residency.ROUTE_SEEN`), for example `mxfp4_k21|le256` or
    `nf4_mtile_captured|gt256`.
  - `prefill_attn` counts each prefill attention call's path (`paged_attention.ATTN_SEEN`): `flash`, or
    `explicit_mask:sinks|window|env`.
  - Both are counted where the route is chosen, in the Python forward: eager calls and graph captures count, graph
    replays do not. An MXFP4 store whose NF4 stacks were freed now shows `mxfp4_*|gt256` instead of `nf4_*|gt256`, which
    `/health` could not tell apart before.
- **Unchanged:** the resolved fields are kept as they were, and no route, kernel or output changes. `tests/test_route_seen.py`
  ties each label to the GEMM that was actually called (mocked, on CPU). Mutating a label or the sinks reason fails it.
