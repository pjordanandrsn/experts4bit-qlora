### P236 registered: grouped-nf4-gemm 0.51.0's decode GEMV against bitsandbytes 0.50.2, op level, on one RTX 5090

A descriptive head-to-head at the released pins, no box yet. Both libraries read the same NF4 bytes (quantized once by
bitsandbytes, block 64, no double quantization): grouped-nf4-gemm's singleton decode call (K33 under `auto`, as
experts4bit-qlora's collapse makes it) against `bitsandbytes.matmul_4bit` (its `gemm_4bit` op, whose own rule picks its
fused kernel or dequantize then `F.linear`), at 1 to 16 rows on Mixtral-8x7B's, Qwen3-30B-A3B's and OLMoE-1B-7B's expert
projections. Per cell: the dequant premise, correctness against an fp32 reference, engagement counted, then 20
palindrome reps. Ratios are reported beside their self-pair bands; nothing is licensed and no default moves.
`bench/p236/` (bench, reducer, runner, driver) and `tests/test_p236.py`, which runs the bench end to end on CPU.
