### `E4B_NF4_SMALLM_BW`: 2..8-token steps on the bandwidth GEMV, opt-in (serving roadmap lever B, step B0)

On the NF4 store's all-resident collapse, `E4B_NF4_SMALLM_BW=1` sends a call of 2 to 8 tokens to singleton groups, one
group per (token, slot) row, as T == 1 already is. grouped-nf4-gemm then serves it through K33's bandwidth GEMV instead
of the device tile table and K25. It is off by default (also when unset), and no speed is claimed: a registered
microbench decides whether it beats K25, and whether a multi-row GEMV is worth building.

- **Per-row identity:** K33 reads one row per program, so each row at 2..8 tokens is bitwise that row at one token
  (`tests/test_nf4_smallm_bw_gpu.py`, run on a GPU box). Rows that share an expert each read its weights.
- **Tokens differ from K25's** (a different function within the tolerance contract), so a default needs a quality bar.
- **Refused, never dropped:** beside `E4B_NF4_GROUPED_SMALLM=1`, on an int4 or MXFP4 store, on a layer not on the
  all-resident collapse, on a kernel package without K33, and at any expert shape where K33 does not engage on the
  device (`GNF4_GEMV_BW`'s rule). `hot_residency.resolve_nf4_smallm_bw` runs at engine build, before capture.
- **Reported:** `/health`'s `nf4_smallm_bw`, and `prefill_routes()["seen"]["moe_nf4_smallm_bw"]` (calls by token
  count). gate_up reads the token rows (`gather_div=top_k`) on this route, as at T == 1.
- **Tests:** `tests/test_nf4_smallm_bw.py` (CPU: routing, refusals, the token-row read, the resolver) and
  `tests/test_nf4_smallm_bw_gpu.py` (GPU: K33 only, per-row bitwise vs T == 1, graph replay, T = 9 and off unchanged).
