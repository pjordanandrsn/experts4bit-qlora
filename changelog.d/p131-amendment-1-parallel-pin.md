### P131 Amendment 1 (#846): lanes rent in parallel, and hosts are pinned under the market rule (registration only)

- **Why.** P131's 10-10 terms made it wait for DQ11's close and SC2f's teardown, with a per-run ACK naming the host.
  SC2f's bar hosts are scarce, so that order held P131 back indefinitely, although P131 has no host-quality bar of its own.
- **The change.**
  - P131 rents in parallel under the standing spending rule.
  - Hosts are pinned like every other lane: up to three stable RTX 5090s at <= $0.85/h, with no per-run ACK.
  - While SC2f's loop is live, SC2f's ten bar machines are excluded.
  - The launch commit is `c7a40682`, with grouped-nf4-gemm at `57f53e19`.
- The arms, rule, predictions, STOP rules, guards and the $5.50 lane ceiling are unchanged.
