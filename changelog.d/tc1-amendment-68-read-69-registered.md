### TC1 amendment 68 read; amendment 69 registered

- **Amendment 68** (`tc1-5090-136`, RTX 5090, $1.96, every arm profiled): on packed rows Unsloth spends 1.160× e4b's GPU time per step
  (P200 HELD; 1.141 on amendment 66's host), runs 7.09× its CPU ops (P202 HELD) and keeps its GPU busy for 0.21 less of each step (P201
  HELD). The packed position is now quoted as that device-time ratio, with the wall ratio per host beside it. P203 FALSIFIED as registered:
  its absolute device-time bound read a card slower for both frameworks. Register row
  `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio`.
- **Amendment 69** registered: token `qwen3pos69` re-reads the field recipe's same-stack position (2.352, before the reentrant checkpoint)
  at the new defaults, every arm profiled, with every bound a ratio within the box (P205-P209). The reducer adds the family, `pos69_why` and
  `score_pos69` (self-test 135).
