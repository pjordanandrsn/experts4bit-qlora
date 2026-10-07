### TC1 amendment 57 registered: the memory census of the training phase on packed rows (P149-P152)

- Token `qwen3memc4kt` runs amendment 47's census box with `TC1_EVAL_EVERY` above `TC1_STEPS`, so no evaluation sits inside the census
  window. The family refuses to run otherwise. Arms: e4b with the fp32 absmax, e4b with it double-quantized, and Unsloth.
- P149: at least 90 % attributed. P150: e4b fp32 2.0-4.5 GB above Unsloth. P151: e4b absmax-dq at most 2.5 GB above. P152: every census
  peak falls in a training step.
- The census scorer takes an optional training-phase prediction (`MEMC_SPECS["train_phase"]`); amendment 47 and 55 readings are
  unchanged. Self-test 122.
