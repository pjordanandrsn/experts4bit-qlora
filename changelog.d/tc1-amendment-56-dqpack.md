### TC1 amendment 56 registered: the double-quantized absmax as a library default on packed rows, with each run's peak split by phase (P144-P148)

- `tc1_arm.py --phase-peaks 1` records the run's peak allocated by phase (setup / eval / train) in `peak_vram_gb_phases`, resetting the
  allocator's max at each boundary. `peak_vram_gb` is unchanged. The flag is refused with `--mem-census 1`. Unit-tested.
- Token `qwen3dqpack` runs e4b's matched arm with the fp32 against the double-quantized expert absmax on packed rows, two draws each, with
  Unsloth's matched arm beside it, every arm with phase peaks.
- P144: dq / fp32 s/step <= 1.02. P145: run peak -1.2 GB or more. P146: held-out within 0.005. P147: e4b's training-phase peak within
  1.0 GB above Unsloth's. P148: e4b's evaluation peak exceeds its training peak.
- The reducer adds the family, `dqpack_why` and `score_dqpack`. Self-test 121.
