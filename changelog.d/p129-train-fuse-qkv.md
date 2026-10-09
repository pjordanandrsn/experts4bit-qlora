### `E4B_TRAIN_FUSE_QKV=1`: one fused q/k/v projection for training attention (opt-in, P129)

Each eligible attention module's q, k and v (`LoRALinear` around bitsandbytes NF4 `Linear4bit`) can run as one fused projection:
- the packed NF4 rows concatenated, with the nested absmax expanded once to fp32, so the fused dequantize is bit for bit the three
  stacked;
- one cast and one LoRA-A matmul over the concatenated A, then three B matmuls;
- serving's fused attention forward (`qkv_fuse._fused_forward`).

The adapters stay the parameters, under their own names. Anything that does not match is refused and keeps today's path
(`TRAIN_QKV_STATS["refused"]` says why). It is off by default: unset, `enable_fast_train` runs today's attention. Serving is not changed.

Memory: the q/k/v NF4 bases are released, and the expanded fp32 absmax adds about 3 bytes per 64-element block (roughly 24 MB on
Qwen3-30B-A3B). A released projection called on its own raises a `RuntimeError` naming the fusion, not a `TypeError` on `None`.

P129 Phase 1, on an RTX A2000 with a two-layer Qwen3-MoE at Qwen3-30B-A3B's layer dimensions:
- 110 fewer kernel launches and 12 % fewer Python calls per training step;
- the projection within TC1's rounding bar;
- in Amendment 1, a 30-step run's divergence inside the eager path's own floor.

Phase 2 measured lower wall time but read QUALITY_FAIL under the registered step-0 bar; the knob remains opt-in.
Amendment 3 registers the calibrated re-measure.
