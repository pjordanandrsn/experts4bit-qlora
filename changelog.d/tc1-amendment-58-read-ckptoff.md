### Read: TC1 amendment 58 -- checkpoint inputs in host memory take 0.74 GB off e4b's packed-row training peak for 0.3 % of the step (P153-P155, P157 HELD; P156 FALSIFIED)

- `tc1-5090-115` ($0.82, a Ryzen 9 7900). With `E4B_CKPT_OFFLOAD=1`, e4b's training-phase peak falls 26.59 → 25.85 GB at 1.003 of the
  step, and held-out moves by +0.0002. Row `e4b.train.ckpt-offload.packed-4k.5090.2026-10-07`.
- Against Unsloth's training-phase peak: +1.72 GB without the offload (P156 missed its 1.7 by 0.02; #1296 took 0.20 GB off the training
  peak) and **+0.98 GB** with it (P157).
- With the offload, e4b's run peak is the held-out evaluation's 26.88 GB. Next, by the registered rule: the offload at the field recipe,
  before any default.
