### P134 registered (Track B1, R-a): does `GNF4_GEMV_BW=1` decode the default `serve_paged` server faster at one request on Granite-3.1-3b-a800m and OLMoE-1B-7B on an RTX 5090?

- `bench/p134/` adds the lane's runner, driver, box and reducer, derived from P133's, with P133's dry fixture.
  `PREREG-p134.md` registers it before any box.
  - Two families run one after the other on one box, each with its own fit check, ABBA arms, quality phases and
    verdict. The arms are `GNF4_GEMV_BW` unset (the scalar GEMV at both families' shapes) against `=1` at the shipped
    plans.
  - The premise is the 5090's: 35 + 18 passed, none skipped.
  - The proof reports each family's g1 band from its own B0 step. OLMoE's band is posted before the reading.
  - The driver refuses any run id but `p134-5090-prove-1`, `p134-5090-1` and `p134-5090-2`.
- `tests/test_p134_staged_pin.py` pins the staged files, the self-tests, the pins, the families, the run ids, the order,
  the guards and the exit codes.
- Nothing in the package changes.
