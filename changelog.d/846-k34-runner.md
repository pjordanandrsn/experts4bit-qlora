### Lane K34's runner (#846): the K16 32/64-row plan census's box side (bench and tests only)

- **What.** `bench/k34/` drives grouped-nf4-gemm's lane K34 (`kernel/PREREG-k34-k16-wide-plan-census.md`, gnf4 #525) on
  one RTX 5090. It is K33's runner with the tripwire, the premise and the bench replaced, and it fetches no model. The
  census is exploratory: a plan it names licenses nothing until its own confirmatory read in this repository.
- **The box.** Refusals (card class, disk) come before the install of gnf4 at `GNF4_SHA`. The tripwire proves two things:
  - the installed commit carries #522's 32- and 64-row tiles;
  - `k34_bench`'s shipped plan is the one `Int4Linear` serves (`plan_smallm` at both shapes, with the signature's warps
    and stages).

  The premise is `kernel/test_int4_smallm_interp.py` compiled on the card, 25 passed and none skipped (rc 23). Then
  `k34_bench.py` runs from the clone at `GNF4_SHA`. Every GNF4 int4 knob starts unset.
- **Tests.** `tests/test_k34_staged_pin.py`: the runner's pin, its shape and order, the exit codes, and the drive's dry run.
