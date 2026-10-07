### TC1 amendment 60 read: the held-out loss from the logits in chunks takes 4.37 GB off e4b's packed-row evaluation peak (P162-P165 HELD)

- `tc1-5090-121` (RTX 5090, AMD EPYC 9755, torch 2.12, $1.46): with `E4B_CKPT_OFFLOAD=1`, `E4B_CHUNKED_EVAL_LOSS=1` lowers the evaluation-phase
  peak from 26.877 to 22.504 GB on both draws, below the 25.85 GB training phase, so e4b's run peak on packed rows is the training phase's,
  0.99 GB above Unsloth's 24.86. Step-0 held-out is identical and held-out at N moves +0.00005.
- Register row `e4b.train.chunked-eval-loss.packed-4k.5090.2026-10-07`, a README section, STATUS, the receipts. By the amendment's rule the
  switch goes to a default PR next.
