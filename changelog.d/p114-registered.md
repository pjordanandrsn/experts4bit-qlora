### P114 registered: the two energy harnesses on a rented RTX 5090, to supersede the A2000 energy rows (#1133 decision 1; bench and tests only)

- `bench/p114/PREREG-p114.md`: three passes each, unchanged and sha-pinned, of:
  - `bench/_upstream/bench_energy.py`: one OLMoE-dims gate_up projection; native bf16, dequant → linear and
    `bnb.matmul_4bit` at decode, prefill and train;
  - `bench/bench_energy_excluded.py`: Part B, J/token as batch grows.

  The stack is bitsandbytes 0.50.2 on one rented RTX 5090. The run is refused unless the card's `power.draw` reads,
  every pass is logged by `nvidia-smi pmon`, and a correctness gate runs first.
- The rule (`bench/p114/p114_reduce.py`, 10-case self-test) reads READ, NOISY (any ratio spreads more than 0.10 over the
  three passes) or VOID.
- On READ, a 5090 row supersedes `e4b.train.energy-honest.scoped-a2000` and
  `e4b.train.energy-honest.a2000-bnb0502.2026-10-04`, and the README's energy sentence is restated from the 5090 medians,
  whatever their sign. No band is borrowed from the A2000.
- `tests/test_p114_lane.py`: the pins, the gate-before-passes order, the exact card check, the guard fitting every alarm,
  and the driver's dry run.
