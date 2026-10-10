### P130 read (RTX 5090): both prefill knobs licensed on SC2e's int4 stack (#846; bench only)

- **`E4B_PREFILL_LEAN_DISPATCH` (P2):** bit-identical on every window. It is ×0.968 [0.966, 0.970] on the served
  first-chunk prefill graph, inside the registered ×1.01 no-slower margin.
- **`E4B_FUSE_PREFILL_GLUE` (P1):** ×1.118 [1.116, 1.119] faster, and at parity on P117's teacher-forced instrument
  (bias −0.0035 nats, inside the floor's bar 0.0125).
- **Both together** take the 512-token first-chunk forward from 38.17 to 32.99 ms on this host. The glue map's ceilings
  were 6.80 ms; the knobs saved 5.18.
- **The defaults do not move here.** Each flip is its own PR with a $0 A2000 smoke, in the next release wave. P1's flip
  is scoped to `qwen3_moe`.
- `bench/p130/RESULTS-p130.md` and `bench/p130/receipts/p130-5090-1/` (with `SHA256SUMS`). The lane cost $1.961 of its
  $5.50 ceiling.
