### TC1 amendment 37 registered: the compact padded LoRA delta again, with grouped-nf4-gemm#473's backward, on another host (bench and tests only)

- **Why.** Amendment 36 found the compact delta faster (0.969 / 0.970) but its matched peak 0.229 GB higher. Its backward held the padded
  output gradient while rebuilding the input block; grouped-nf4-gemm#473 releases each intermediate at its last use (values identical),
  and on an RTX A2000 its per-call backward peak went from 24–55 % above the autograd path's to 2–30 % below.
- **The box** (token `qwen3compactab2`): amendment 36's box with grouped-nf4-gemm at or after #473, off machine 145701. P73 matched
  peak change in [−0.05, +0.50] GB of drop; P74 / P75 speed in [0.95, 0.99]; P76 held-out within 0.005. All four HELD makes it
  grouped-nf4-gemm's default.
- The reducer reads it with amendment 36's scorer, now per family (`COMPACT_SPECS`); one new self-test case. Amendment 36's own
  receipts re-reduce to the same verdicts.
