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
- **The q/k/v bases stay in the model.** The fusion used to release them, so while fused a full `state_dict` had no q/k/v base
  weights (a Trainer checkpoint, `save_pretrained`), a resume met unexpected keys, a direct call to a projection raised, and
  `disable_fast_train` left the attention fused. Under the opt-in that reached only runs that asked for it; as the default it would
  reach every Qwen3-MoE trainer, and every enable-then-disable caller, among them TC1's and tp4's `attn_only` arms, which probe
  `enable_fast_train` before training without it. Now:
  - each base is re-pointed at the fused copy: its packed weight is a view of the fused bytes, and its quant state is a non-nested
    one over its slice of the expanded fp32 absmax, which dequantizes bit for bit as the nested statistics did. Nothing is
    duplicated, and a device move re-points the bases at the moved bytes;
  - `state_dict` carries every q/k/v weight. The base quant state is the non-nested form, so `weight.nested_absmax` and
    `weight.nested_quant_map` are absent for those bases. A checkpoint saved while fused therefore stores the q/k/v absmax in fp32,
    not nested; bitsandbytes rebuilds it as a plain quant state, which dequantizes bit for bit. Resumes load with `strict=False`
    (torch's strict check refuses every bitsandbytes 4-bit dict, an unfused one included): a fused model's dict loads into an
    unfused model, and a resume into a fused one writes through the views;
  - a direct call to a projection computes what it computed before the fusion;
  - `disable_fast_train` gives the attention its own forward back, the three projections compute what they did before, and a
    second `enable_fast_train` fuses again. The 3 bytes per 64 values stay until the model is reloaded.
- **Memory.** The training estimate prices the fused projection's fp32 absmax: 3 bytes per 64 fused q/k/v values, about 24 MB on
  Qwen3-30B-A3B. The boxes measured a peak change of +0.005 / +0.023 GB. `estimate_env()` reports the knob.
