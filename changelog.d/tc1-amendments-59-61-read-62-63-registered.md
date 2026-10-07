### TC1 amendments 59 and 61 read; amendments 62 and 63 registered

- **Amendment 59** (`tc1-5090-120`, RTX 5090, Ryzen 9 9950X, $1.37): at TC1's field recipe, keeping checkpoint inputs in host memory steps
  0.948 (matched) and 0.916 (shipped) of the default, takes 0.171 GB off the matched training peak, and leaves held-out within 0.003
  (P158-P161 HELD). Register row `e4b.train.ckpt-offload.field.5090.2026-10-07`.
- **Amendment 61** (`tc1-5090-122`, RTX 5090, EPYC 9655, $1.40): the combine over row chunks steps 0.978 of the whole-tensor combine with
  held-out unchanged (P167, P168 HELD) but leaves the packed-row training peak where it was (-0.009 GB; P166 FALSIFIED), and e4b's
  training phase 0.979 GB above Unsloth's (P169 FALSIFIED). Register row `e4b.train.combine-row-chunks.packed-4k.5090.2026-10-07`.
- **Amendment 62** registered: token `qwen3ckptre` reads the shipped arm at the field recipe with Hugging Face's checkpoint, the reentrant
  checkpoint alone (`E4B_CKPT_OFFLOAD=reentrant`) and with its inputs in host memory, to say which made the step faster (P170-P173).
  `tc1_arm.py` records `ckpt_offload_funcs`; the reducer adds the family, `ckptre_why` and `score_ckptre` (self-test 127).
- **Amendment 63** registered: token `qwen3ckptoff28` reads amendment 59's A/B in the field image's torch 2.8 with every arm profiled.
  Its speed predictions (P174, P175: g1 / g0 at most 1.01) are scored only on a host-bound host: the g0 arm's device busy fraction
  against its timed step must be at most 0.9. P176 checks held-out and P177 the training peak. Amendment 59's rule requires this read
  before the offload can become a default. The reducer adds the family and `score_ckptoff28` (self-test 128).
