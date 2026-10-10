### `E4B_PREFILL_LEAN_DISPATCH` defaults to `auto`: K19's lean dispatch on prefill rows (#846, P130)

- K19's prefill rows (the int4 route, above 256 routed rows) now take K23's dispatch by default. gate_up reads the token
  rows through `gather_div`, and down stores in the caller's order through `scatter=order`. The 4,096-row gather and
  the unsort are gone, and the chained 16-row tile table is kept.
- **The read** (`bench/p130/RESULTS-p130.md`, one RTX 5090, Qwen3-30B-A3B int4): bit-identical on every window of the
  served first-chunk prefill graph, and ×0.968 [0.966, 0.970] per 512-token replay (1.15 ms of 38.2).
- **`auto`** takes it wherever the installed K19 takes `scatter=` / `gather_div=`, and keeps the gather on an older
  kernel package. `1` refuses such a package; `0` restores the gather and unsort. `prefill_routes.prefill_lean_dispatch`
  reports the setting (`auto`, `1` or `0`).
- Register row `e4b.serve.p130.prefill-lean-dispatch.qwen3.5090.2026-10-10`. Other families on the int4 K19 prefill
  route take it on its bitwise contract, not on a read of their own.
