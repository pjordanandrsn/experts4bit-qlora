### TC1 amendments 59 and 61 read; amendment 62 registered

- **Amendment 59** (`tc1-5090-120`, RTX 5090, Ryzen 9 9950X, $1.37): at TC1's field recipe, keeping checkpoint inputs in host memory steps
  0.948 (matched) and 0.916 (shipped) of the default, takes 0.171 GB off the matched training peak, and leaves held-out within 0.003
  (P158-P161 HELD). Register row `e4b.train.ckpt-offload.field.5090.2026-10-07`.
- **Amendment 61** (`tc1-5090-122`, RTX 5090, EPYC 9655, $1.40): the combine over row chunks steps 0.978 of the whole-tensor combine with
  held-out unchanged (P167, P168 HELD) but leaves the packed-row training peak where it was (-0.009 GB; P166 FALSIFIED), and e4b's
  training phase 0.979 GB above Unsloth's (P169 FALSIFIED). Register row `e4b.train.combine-row-chunks.packed-4k.5090.2026-10-07`.
- **Amendment 62** registered: token `qwen3ckptre` reads the shipped arm at the field recipe with Hugging Face's checkpoint, the reentrant
  checkpoint alone (`E4B_CKPT_OFFLOAD=reentrant`) and with its inputs in host memory, to say which made the step faster (P170-P173).
  `tc1_arm.py` records `ckpt_offload_funcs`; the reducer adds the family, `ckptre_why` and `score_ckptre` (self-test 127).
