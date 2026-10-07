### P116 registered: does grouped-nf4-gemm K33's bandwidth-targeted NF4 decode GEMV (`GNF4_GEMV_BW=1`) decode the default `serve_paged` server faster at one request, at no measurable quality cost?

- `bench/p116/` adds the lane's runner, driver, box, reducer and pins. `PREREG-p116.md` registers it before any box.
  - The arms are B0a B1a B1b B0b on the default graph server (Qwen3-30B-A3B NF4, one RTX 5090). Only grouped-nf4-gemm's
    decode GEMV switch differs: unset against `GNF4_GEMV_BW=1` at K33's selected plans (`GNF4_GEMV_BW_PLAN`).
  - The served server reaches the switch only at T == 1, so W1 is the subject and W16 a no-regression control.
  - Quality is P110's teacher-forced bar at one window per pass. That is the served W1 arithmetic, with floors from
    `chunk` and `rep`.
  - Engagement is grouped-nf4-gemm's dispatch tally: B1 all `bw_prmt32`, B0 dot-pad.
  - DEFAULT_ON iff g1 ≥ 1.03, g16 ≥ 0.99 and the quality bar holds.
- `tests/test_gemv_bw_served_gpu.py` (GPU) pins the served route on a tiny all-hot NF4 store under the served collapse.
  T == 1 dispatches only `bw_prmt32` with the switch on and none with it off. Both routes read the reference. The T == 1
  step captures and replays bitwise as eager. T > 1 never reaches the decode GEMV.
- `tests/test_p116_staged_pin.py` pins the staged files, the self-tests, the order, the subject, the guards and the exit
  codes.
- Nothing in the package changes; every default is as before.
