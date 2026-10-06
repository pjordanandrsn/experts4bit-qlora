### TC1 amendment 51 registered: the packed 4,096-token position on one stack at e4b's defaults (P126-P129)

- Token `qwen3samestack4kd`: amendment 43's packed same-stack box with nothing set. The chunked LM loss (`auto`, #1203) and
  grouped-nf4-gemm's bucketed padding (`auto`, grouped-nf4-gemm#492) both engage on these rows by themselves, and validity checks that they
  did. P126: Unsloth/e4b in [1.25, 1.80]; P127: environment in [0.84, 0.98]; P128: every e4b arm resident; P129: e4b's matched peak <=
  29.5 GB. Bands set with amendments 43 and 48 in view. A stable P126 with P128 becomes the default-settings packed position, superseding
  amendment 39's out-of-memory row as that reading.
- `tc1_reduce.py`: the family, `packed_defaults_why`, `score_packed4kd_peak`; self-test 116 cases.
