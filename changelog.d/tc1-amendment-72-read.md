### TC1 amendment 72 read: the single-block ladder's `auto` becomes grouped-nf4-gemm's default

- **Amendment 72** (`tc1-5090-141`, RTX 5090, AMD Ryzen 9 9950X3D, $0.80, GPU-bound, no load voids): amendment 71's box
  on a third host. With `NF4_QLORA_SINGLE_LADDER=auto` the fp32-adapter arm stepped 1.031, inside the registered 1.05,
  and the bf16 arm 1.002. `aten::bmm`'s CPU time per call fell to 0.092, device time rose 5.5 % on the fp32 arm, and
  held-out was unchanged (P215–P221 HELD). By the rule `auto` becomes grouped-nf4-gemm's default (grouped-nf4-gemm#521).
  Register row `e4b.train.single-ladder-auto.gpu-bound.5090.2026-10-08`.
