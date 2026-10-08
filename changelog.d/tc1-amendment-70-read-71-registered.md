### TC1 amendment 70 read; amendment 71 registered

- **Amendment 70** (`tc1-5090-138`, RTX 5090, AMD EPYC 7K62, $1.73, every arm profiled): grouped-nf4-gemm's opt-in single-block ladder
  (`NF4_QLORA_SINGLE_LADDER=1`) steps e4b's matched arm (fp32 adapters) 0.797 at the field recipe on a host-bound box. `aten::bmm`'s CPU
  time per call falls from 305 µs to 24.5 µs, for 4.1 % more device time and 0.33 GB more peak (P210, P212-P214 HELD). The shipped arm
  (bf16 adapters) steps 1.015 (P211 FALSIFIED), so the flag stays opt-in. Register row `e4b.train.single-ladder.field.5090.2026-10-08`.
- **Amendment 71** registered: token `qwen3slauto` reads `NF4_QLORA_SINGLE_LADDER=auto` (grouped-nf4-gemm#514, the ladder exactly when the
  adapters are fp32) against 0 on a second host (P215-P219). If P215, P216 and P219 hold, `auto` becomes grouped-nf4-gemm's default. The
  reducer adds the family, `slauto_why` and `score_slauto` (self-test 137).
