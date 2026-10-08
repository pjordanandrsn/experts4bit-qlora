### P115 Amendment 3 registered (#1313): Phase D, the fused stack on top of P116's bandwidth GEMV, and the family-scoped default

- **Why.** Phase C read FLIP_HELD, so any default must be family-scoped under a new registration. grouped-nf4-gemm
  0.43.0 made P116's bandwidth decode GEMV the default, and the fused stack and that GEMV were never read together.
- **What.** `bench/p115/p115d_{run,drive,box,reduce}` and `staged-d.sha256`, plus `tests/test_p115d_staged_pin.py`.
  - **Subject:** Qwen3-30B-A3B NF4 on the default server at 16 slots, grouped-nf4-gemm v0.43.0, the GEMV at its default.
  - **Arms:** D0 (the four knobs `0`) against D1 (all `auto`).
  - **Gate:** Phase C's SANE gate (12 windows; |bias| ≤ 0.02 nats, argmax ≥ 0.95), at **one window per pass**
    (T == 1, where the GEMV and the folds meet). The dispatch tally must show the bandwidth route in both phases.
  - **Reported:** the speed of an ABBA W1/W16 pair, the stack's gain on top of the GEMV.
- **Rule.** VOID → FUNCTION_FAIL → COMBINED_FAIL → **COMBINED_SANE**, which licenses the family-scoped default
  (allowlist `qwen3_moe`, `qwen3_5_moe`, `granitemoe`).
- **Budget.** Proof on Granite (guard 0.75 h), reading guard 1.5 h. Phase D ceiling $4.00, inside P115's $10 hard stop
  ($2.455 spent).
- **Mechanism: (B), unset resolves per family; explicit `auto` structural, with a warning off the allowlist** (the
  maintainer's decision). An unset knob resolves to `auto` on `qwen3_moe`, `qwen3_5_moe` and `granitemoe`, and to `0`
  elsewhere; `/health` reports the resolution and its source. Explicit `auto` on any other family logs a one-time
  warning naming the read it lacks or failed.
- **Not measured yet.** No box runs before this page merges. The default change itself is its own PR, after Phase D
  reads.
