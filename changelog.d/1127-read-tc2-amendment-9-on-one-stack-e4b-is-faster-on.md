### Read: TC2 amendment 9 — on one stack e4b is faster on Mixtral-8x7B too, Unsloth/e4b 1.144 (P29, P30, P31 HELD); it becomes Mixtral's quoted position

- `tc1-5090-84` ($1.59, EPYC 7B13, 60-step load-gated draws): Mixtral resident at e4b's defaults (the dense route), both frameworks on
  torch 2.12.1 / transformers 5.5.0. e4b 3.233 / 3.248 s/step, Unsloth 3.701 / 3.711: **1.144** [1.140, 1.148], COMPARABLE. e4b on the
  field image's stack in the same box: 3.681 / 3.636, so the environment reads **0.886**, and the dense route ran on every e4b arm.
- Amendment 8's 0.836 (Unsloth faster, on a 285K host with e4b on torch 2.8) stays as that reading. On this host e4b on its own stack
  reads 1.013; one stack moves it to 1.144. Unsloth keeps a 2.07 GB lower peak at e4b's defaults (the fp32 absmax) and ×0.93 the energy.
