### The fused q/k/v training projection is on by default (P129 DEFAULT_ON)

- **What changes.** `enable_fast_train` now fuses each eligible attention module's q, k and v (NF4 base, LoRA adapters) into one
  projection by default. `E4B_TRAIN_FUSE_QKV=0` (or `false` / `off` / `no`) keeps the three projections, op for op as before.
- **Where it applies.** Only where the knob already applied: Qwen3-MoE attention whose q/k/v are `LoRALinear` around matching NF4 bases.
  Everything else is refused and unchanged.
- **Measured** on Qwen3-30B-A3B on RTX 5090, two hosts, at TC1's field recipe:
  - launches per step −13.9 % (fp32 adapters) and −14.6 % (bf16);
  - the step at 0.895 / 0.868 on the first host and 0.906 / 0.884 on the second;
  - device time 0.987–0.988;
  - the step-0 held-out inside the box's fp32-anchored rounding envelope, and held-out at N within 0.002.

  Claims: `e4b.train.p129.fused-qkv.qwen3.5090.2026-10-10`, `e4b.train.p129.fused-qkv.second-host.5090.2026-10-10`.
- **`disable_fast_train` undoes it.** Each projection gets its base back as a view of the fused bytes, with its slice of the
  expanded fp32 absmax, which dequantizes bit for bit as the nested statistics did. The three projections then compute what they
  computed before the fusion, and a second `enable_fast_train` fuses again. Before this change, a disable left the attention fused
  with its bases released. Under the opt-in that reached only runs that asked for it; under the default it would reach every
  enable-then-disable caller, among them TC1's and tp4's `attn_only` arms, which probe `enable_fast_train` before training without
  it. The 3 bytes per 64 values stay until the model is reloaded.
- **Memory.** The training estimate prices the fused projection's fp32 absmax: 3 bytes per 64 fused q/k/v values, about 24 MB on
  Qwen3-30B-A3B. The boxes measured a peak change of +0.005 / +0.023 GB. `estimate_env()` reports the knob.
