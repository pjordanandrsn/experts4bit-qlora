### TC1 amendment 70 registered: grouped-nf4-gemm's single padded block on its ladder at the field recipe

- Token `qwen3sladder`: `NF4_QLORA_SINGLE_LADDER` 0 against 1 (grouped-nf4-gemm#513) at TC1's field recipe, e4b's shipped and matched arms,
  two draws a side, every arm profiled. P210 / P211 predict s/step at most 0.97 on a host-bound box; P212 `aten::bmm`'s CPU time per call
  halved; P213 device time at most 1.05; P214 held-out unchanged. If P210, P211, P213 and P214 hold, it becomes grouped-nf4-gemm's default.
- `tc1_arm.py` records grouped-nf4-gemm's `SINGLE_LADDER_STATS` (`single_ladder`). The reducer adds the family, `sladder_why` and
  `score_sladder` (self-test 136).
