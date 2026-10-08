### TC1 amendment 69 read

- **Amendment 69** (`tc1-5090-137`, RTX 5090, AMD EPYC 7713, $2.12, every arm profiled): at TC1's field recipe on one stack, with e4b at
  every current default, Unsloth spends 1.924× e4b's GPU time per step, and the wall-clock ratio is 2.803 on that host (P205-P209 HELD).
  Unsloth runs 14.1× e4b's CPU ops per step, and its GPU is busy for 0.33 of its step against e4b's 0.49. STATUS's Qwen3-30B-A3B
  position to quote is now the GPU-time ratio, with the host's wall ratio beside it. Register row
  `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio`.
