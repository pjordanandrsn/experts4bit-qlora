### TC1 amendments 62 and 63 read; amendment 64 registered

- **Amendment 62** (`tc1-5090-127`, RTX 5090, EPYC 9655, $1.11): the field recipe's speed-up is the reentrant checkpoint's. Alone it steps
  0.900 of Hugging Face's checkpoint on the shipped arm, and the host-memory copies cost 1.019 on top; held-out within 0.0011 (P170-P173
  HELD). Register row `e4b.train.ckpt-flavour.field.5090.2026-10-07`.
- **Amendment 63** (`tc1-5090-125`, RTX 5090, Ryzen 9 9950X, $1.48): in torch 2.8 on a host-bound box (premise met) the offload steps
  0.994 (matched) / 0.985 (shipped) and takes 0.181 GB off the matched training peak (P174-P177 HELD). Register row
  `e4b.train.ckpt-offload.field-torch28.5090.2026-10-07`.
- **Amendment 64** registered: tokens `qwen3ckptre4k` (packed rows, torch 2.12, matched arm) and `qwen3ckptre28` (field recipe, torch 2.8,
  shipped arm) read the three checkpoints where amendment 62 did not: P178-P183, toward the reentrant checkpoint as e4b's default
  checkpoint. The reducer adds the families and `score_ckptre64` (self-test 130). Added in review: every torch 2.8 arm is profiled and P182 is read only
  on a host-bound box (amendment 63's premise gate).
