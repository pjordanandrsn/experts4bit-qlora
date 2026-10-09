### P129 Phase 2 read (#835): QUALITY_FAIL, as registered; the cause is GEMM-shape rounding

- **The box** (`tc1-5090-146`, one RTX 5090). The fused q/k/v training projection cut launches per step by 13.9–14.6 %. It stepped at
  0.896 (matched) and 0.879 (shipped) of the knob-off time on a host-bound host, with device time 0.985–0.987.
- **Why it failed.** The step-0 held-out loss moved by −0.01233 on both arms, against TC1's 0.0005 bar. Held-out at N held. By the
  rule nothing turns on.
- **The investigation** (RTX A2000, the real model at the pin, `bench/p129/instrument/p129_step0.py`):
  - the fused module with serving's forward equals the stock forward fed the same q/k/v, bitwise in every layer;
  - the shift is the base projections' GEMM shape, amplified through 48 layers;
  - a neutral split of q's matmul moves the step-0 loss as much, and the sign differs between cards.
- **The bar.** The tested fused and neutral GEMM-shape changes exceeded the original step-0 bar; a re-measure uses an
  amendment with a floor measured on the box.
