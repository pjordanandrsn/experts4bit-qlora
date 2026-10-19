### Added

- **P153's W16 cost: token divergence bounded out** (#1313). A $0 router replay on the A2000, recorded under
  `bench/p153/router-replay/`, bounds the token-divergence term out:
  - **The method:** S0's and S1's recorded W16 streams went through the same Qwen3-30B-A3B, teacher-forced, in bf16.
  - **The pre-stated metric:** distinct experts per counted bucket-16 step, S1/S0. It reads 0.9990 against bands stated
    before the run, which reject anything at or below 1.
  - **Row spread:** in a robustness check suggested at review, every row-per-expert spread metric is within +/-0.5 %.
  - **What remains:** the 16-request cost stays observed, not attributed. The resident int8 copy is the remaining named
    candidate, and testing it is a paid served A/B, registered first. No default changes.
