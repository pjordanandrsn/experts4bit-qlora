### TC1 amendment 75 registered: the 4090-fit read, with the compact delta, the chunked LM loss and both, reserved and device peaks first

- One rented RTX 5090 at TC1's field recipe with the memory census, e4b only at today's pins (main at launch, grouped-nf4-gemm 0.49.0).
  The arms are e4b's matched arm at its defaults, with `NF4_QLORA_COMPACT_DELTA=1`, with `E4B_CHUNKED_LM_LOSS=1`, and with both. One
  draw each, worst case $3.20. The recipe is fixed; any smaller batch, quantization change or offload would be its own labelled variant.
- Each arm is bound to its delta path and its loss, or it is VOID. The read leads with each arm's reserved and nvidia-smi peaks, beside
  a 4090's 24,564 MiB total. The successor to amendment 74's rule first compares the target's live bytes at both peaks, then credits
  the drop up to the next transient. A candidate also needs the reserved and device peaks to fall, and goes to its own pre-registered
  4090 acceptance box.
- `bench/tc1/tc1_run.sh` (`qwen3pos75cl`), `bench/tc1/tc1_reduce.py` (`pos75_why`, `pos75_rule`, `pos75_block`, `vram_peak_mib`,
  `unlisted_bound`, self-test case 130) and a test of the rule on box 154's committed cells.
