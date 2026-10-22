### HO2 read: the planner's first held-out training rows, COVERED 8 of 8

- Two rented boxes, an RTX 5090 and an RTX 4090, ran `loggetta train` with the planner's own plan, using no receipts on file. There
  were four rows each: LFM2-8B-A1B and ERNIE-4.5-21B-A3B, at 1,024 and 2,048 tokens. On every row, the plan's total bounded the sampled
  driver peak and its allocator estimate bounded the allocated peak (driver / plan 0.806-0.874, allocated / estimate 0.915-0.963).
  USD 0.741 in all.
- Two lines bind the read:
  - The 2,048-token rows ran the chunked LM loss under the auto gate, while the 1,024-token rows ran the stock loss. Each branch is
    inferred, not counted. So the rows are not a length comparison.
  - Every row planned and ran experts on device, so HO2 tested the device placement only. The registration's expected host offload
    for ERNIE on the 4090 did not occur.
- `bench/ho2/receipts/` holds both boxes' identifier-free files and the row receipts with the GPU UUID nulled. `bench/ho2/ho2_public.py`
  derives those receipts and rebuilds each box's verdict byte for byte. `tests/test_ho2_public.py` covers the rebuild, the sums and
  the host-identifier checks.
