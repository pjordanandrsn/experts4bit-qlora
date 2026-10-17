### P133 registered (Track B1, R-c): does `GNF4_GEMV_BW=1` decode the default `serve_paged` server faster at one request on an RTX 4090 (sm_89, 128 SMs), where auto keeps it off?

- `bench/p133/` adds the lane's runner, driver, box, reducer, dry fixture and pins, derived from P116's.
  `PREREG-p133.md` registers it before any box.
  - Arms B0a B1a B1b B0b on the default server (Qwen3-30B-A3B NF4): `GNF4_GEMV_BW` unset (the scalar GEMV at 128 SMs)
    against `=1` at the shipped plan table. W1 is the subject; the control is W16 at the box's own width.
  - Exact pins: e4b `c8d7e629`, grouped-nf4-gemm 0.46.0, torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2.
  - A fit check runs before any arm: if W1 does not fit the 4090, the read is VOID with no rerun.
  - The premise requires exact pass counts and skipped test ids, so a new skip fails.
  - The driver refuses any run id but `p133-4090-prove-1`, `p133-4090-1` and `p133-4090-2`; `P133_DRY=1` is the $0
    dry run on a tiny fixture.
- `tests/test_p133_staged_pin.py` pins the staged files, the self-tests, the pins, the run ids, the order, the subject,
  the guards and the exit codes.
- Nothing in the package changes.
