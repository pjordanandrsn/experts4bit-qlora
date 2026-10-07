### Read: TC1 amendment 56 -- the double-quantized absmax costs packed rows 0.2 % for 1.35 GB; e4b's training phase peaks 1.92 GB above Unsloth's with it (P144-P146, P148 HELD; P147 FALSIFIED)

- `tc1-5090-113` ($1.05, an EPYC 7713), torch 2.12. `E4B_ABSMAX_DQ=1` steps 1.002 of the fp32 absmax's time. It lowers the run peak by
  1.35 GB, and held-out moves by +0.0002. Row `e4b.train.absmax-dq.packed-4k.5090.2026-10-07`.
- Peaks by phase:
  - e4b's training steps peak at 28.14 GB with the fp32 absmax, about 0.09 GB below its evaluation (28.23);
  - with the double-quantized absmax, its training phase peaks at 26.79 GB against Unsloth's 24.86, which is +1.92 GB.
- This corrects how amendment 55's census was read: the packed-row gap is in training, not in the evaluation's logits. STATUS says so.
- Next, by the registered rules: a library PR defaults `enable_fast_train` to the compressed absmax for resident training, and a
  training-phase census.
