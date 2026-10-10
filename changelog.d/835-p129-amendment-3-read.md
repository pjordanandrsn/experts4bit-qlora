### P129 Amendment 3 read (#835): GAIN; the fused q/k/v training projection stays opt-in until a second host reads it

- **The box** (`tc1-5090-147`, one RTX 5090, host-bound). `E4B_TRAIN_FUSE_QKV=1` cut launches per step by 13.9 % (fp32 adapters)
  and 14.6 % (bf16 adapters). It stepped at 0.895 and 0.868 of the knob-off time, with device time 0.987.
- **Quality.** The step-0 held-out lies within the box's fp32-anchored envelope (0.59 of it), and held-out at N is within 0.0007.
- **The caveat.** Every standing attempt ran above the 6.0 load gate, so the wall ratios were read under contention. They match the
  first box's on another host (0.896 / 0.879).
- **Why step 0 moves.** The router's top-8 choice amplifies sub-ulp q/k/v rounding. Pinning the routing removes the shift
  (`bench/p129/records/a3/routing_step0.json`).
- **Files.** Receipts for `tc1-5090-146` and `tc1-5090-147` (with logs and SHA256SUMS), a claims row, and a STATUS line.
