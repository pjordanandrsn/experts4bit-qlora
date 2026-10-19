### P131 results (#846): the split K19 tile table is not faster on the served prefill graph

On one RTX 5090, K19 over a split tile table (`E4B_K19_PREFILL_BIG_M`) computes the 16-row table's arithmetic bit for
bit at both tall tiles, but neither clears the registered x1.02 on the served 512-token first-chunk prefill graph of
Qwen3-30B-A3B: m32 reads 0.9836 and m64 1.0073 over eight processes. The verdict is M32_HELD M64_HELD DEFAULT_NONE, so
the knob stays opt-in and nothing changes by default. Record: `bench/p131/RESULTS-p131.md`, with the proof, a host
stall and the reading's receipts.
