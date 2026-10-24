### Added

- P234 results (`bench/p234/RESULTS-p234.md`): `E4B_NF4_SMALLM_BW=1` (2-8-token decode calls on K33's bandwidth GEMV,
  one group per row) on Mixtral-8x7B's DEFAULT NF4 server, one RTX 5090: **LICENSED**.
  - **Speed:** two requests decode 2.12452x and four requests 1.76481x as fast as the switch unset (min over two
    palindromic pairs).
  - **Unchanged speed:** one request (0.99182, identical tokens) and sixteen requests (1.00013, tokens differ in some
    rows).
  - **Quality:** within P110's bars at group 4.
  - **Records:** register row `e4b.serve.p234.nf4-smallm-bw.mixtral-nf4.5090.2026-10-24`; receipts in
    `bench/p234/receipts/p234-5090-1/`.
  - **What follows:** the default for buckets 2 and 4 lands in its own PR; 5 to 8 tokens stay opt-in.
