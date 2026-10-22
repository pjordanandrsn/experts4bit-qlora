### TC1 amendment 73 read: the field-memory gap is the padded LoRA delta's transients plus the allocator cache

- On one rented RTX 5090 at TC1's field recipe, with the memory census on both arms at the field-device-ratio claim's commits, both
  claim peaks reproduce (e4b 26.156 GB, Unsloth 24.273 GB). Every resident tensor class is the same size at the peak, and the
  1.887 GB gap is transient: 2.044 GB of e4b's is grouped-nf4-gemm's padded LoRA delta. At the end of setup both hold the same
  allocated bytes; e4b's 0.713 GB more reserve is the allocator's cache. Descriptive, one draw each, no speed read.
- `bench/tc1/tc1_public.py`: a TC1 box's public receipt copy. It writes derived copies of the files that name the rented host, without
  those fields, and rebuilds the box's RESULTS from the committed files through the registered reducer (`tests/test_tc1_public.py`).
- Claim `e4b.train.memory-census.field.5090.2026-10-22`; README section, STATUS record line, and the receipt's public copy under
  `bench/h2h-2026-10-02/tc1/receipts/tc1-5090-152/`.
