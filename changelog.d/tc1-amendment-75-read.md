### TC1 amendment 75 read: no arm fits an RTX 4090 yet; the compact delta and the chunked LM loss each save a credited amount

- One rented RTX 5090 at TC1's field recipe with the memory census, e4b's matched arm at today's pins, one draw per arm. It read
  today's defaults, `NF4_QLORA_COMPACT_DELTA=1`, `E4B_CHUNKED_LM_LOSS=1` and both, with the reserved and nvidia-smi peaks read first.
- **No arm fits a 24 GiB RTX 4090.** The best arm (both switches) peaks at 25,727 MiB, 1,163 MiB over a 4090's 24,564 MiB total. Even
  with no allocator gap, its allocated peak plus the CUDA context is about 24,735 MiB.
- By the registered successor rule, the compact delta against the defaults and the chunked loss against the compact delta are
  CANDIDATES: credited memory savings, not a fit. At the best arm's peak, the compact delta's own backward holds two 0.811 GB buffers.
- Claim `e4b.train.4090-fit-read.field.5090.2026-10-22`. The README section and STATUS lines are added, and the receipt's public copy
  is under `bench/h2h-2026-10-02/tc1/receipts/tc1-5090-155/`.
