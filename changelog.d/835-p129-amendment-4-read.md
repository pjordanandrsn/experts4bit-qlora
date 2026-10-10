### P129 Amendment 4 read (#835): GAIN on a second host -- DEFAULT_ON is met for the fused q/k/v training projection

- **The box** (`tc1-5090-150`, one RTX 5090, host-bound, no draw voided for load). `E4B_TRAIN_FUSE_QKV=1` cut launches per step by
  13.9 % (fp32 adapters) and 14.6 % (bf16).
  - **Speed:** it stepped at 0.906 and 0.884 of the knob-off time, with device time 0.988.
  - **Quality:** the step-0 held-out lies inside the fp32-anchored envelope, bitwise the first host's numbers, and held-out at N is
    within 0.0019.
- **The decision.** With box 147's GAIN on another host, this meets P129's DEFAULT_ON rule. The default flips in its own change.
- **Files.** Receipts for `tc1-5090-150` with logs and SHA256SUMS, RESULTS-p129-a4.md, a claims row and the STATUS line.
