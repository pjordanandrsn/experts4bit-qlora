### DQ11 MLP-adapter BUILD Phase 2 read (#1083): fused MLP adapters, correct and 2.8% less time per update, NO_GAIN

`dq11-mlp-phase2-1` ran on one RTX 5090 at the science lock and read **NO_GAIN: H 0.0279 < 0.05** (measured-private).
- F1 (`E4B_DENSE_MLP_LORA_FUSE=1`) takes 2.8–2.9% less time per update than F0 at every block pair, with no NOISY
  self-pair and K8 quality unchanged.
- All 33 checked tensors are judged correct by the FP64 bar at the science lock (k ≥ 12 distinct reference paths each).
- The registered H range was right, and the drift prediction was wrong (B 0.123 at update 40), which the shakedown had
  foreshadowed.

The lever closes with its number, and the engine stays opt-in. See `bench/dq11/RESULTS-dq11-mlp-build.md`.
