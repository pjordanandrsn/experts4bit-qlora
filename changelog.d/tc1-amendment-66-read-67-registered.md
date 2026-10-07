### TC1 amendment 66 read; amendment 67 registered

- **Amendment 66** (`tc1-5090-133`, RTX 5090, i9-14900K, $1.58): grouped-nf4-gemm's compact bucketed delta takes 0.654 GB off the matched
  packed-row training peak and steps 0.972 (matched) / 0.977 (shipped) with less device time, held-out unchanged (P189-P192, P194 HELD).
  e4b's matched peak stays 1.044 GB above Unsloth's (P193 FALSIFIED narrowly). By the rule it becomes grouped-nf4-gemm's default
  (grouped-nf4-gemm#508). Register row `e4b.train.compact-buckets.packed-4k.5090.2026-10-07`.
- **Amendment 67** registered: token `qwen3pos67` reads the packed-row position at all the new defaults: e4b defaults, e4b with the offload
  and Unsloth, two draws each (P195-P199). The reducer adds the family, `pos67_why` and `score_pos67` (self-test 133).
