### TC1 amendment 65 read; amendment 66 registered

- **Amendment 65** (`tc1-5090-130`, RTX 5090, Ryzen 9 9950X, $1.26): at the new defaults e4b's packed-row training phase peaks +1.717 GB
  over Unsloth (+0.962 with `E4B_CKPT_OFFLOAD=1`), all of it transient; every static class is equal. With the offload, the largest
  transients at the peak are grouped-nf4-gemm's bucketed LoRA delta, about 1.70 GB in four groups at three source lines (P184-P188 HELD). Register row
  `e4b.train.memory.packed-4k-train-census-defaults.5090.2026-10-07`.
- **Amendment 66** registered: token `qwen3cbk` reads grouped-nf4-gemm's `NF4_QLORA_COMPACT_BUCKETS=1` (#505, the bucketed delta as one
  autograd node, the same bytes) against the autograd body on packed rows, shipped and matched arms, Unsloth beside (P189-P194; P194,
  added by the maintainer before any box, reads the node's device time per profiled step, which a host-bound box's s/step can hide).
  `tc1_arm.py` records `compact_buckets`; the reducer adds the family, `cbk_why` and `score_cbk` (self-test 132).
