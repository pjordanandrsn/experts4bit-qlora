### DQ11 MLP-adapter BUILD registered before code or data (#1083)

The L-census read `BUILD_MLP`, so `bench/dq11/DQ11-MLP-BUILD-PREREG.md` registers its BUILD:
- a dense gate/up/down LoRA fusion engine, off by default and bitwise off;
- an A2000 correctness phase against the unfused path's own neutral floor;
- a matched unfused/fused update on an RTX 5090, read as H = the minimum over four ABBA pairs, at ≥ 5%, with K8
  unchanged.

The registration states a ceiling of about 9% and predicts NO_GAIN as the likelier outcome. No default changes.
