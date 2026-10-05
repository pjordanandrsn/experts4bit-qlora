### `enable_dense_offload` keeps a trainable parameter resident beside frozen ones and warns on an unfrozen model (behaviour change; #1165)

- **The crash.** On a PEFT/QLoRA model, the trainable LoRA matrices over `MIN_BYTES` were selected for streaming. At
  25600 wide, `lora_B` is 1.6 MB. Eviction swapped them for empty placeholders, and `AdamW.step` raised "The size of
  tensor a (0) must match the size of tensor b (16)". Found by the DQ3 rehearsal (#1083) at Qwen3-32B width.
- **The new selection is decided per call.** If any streamable parameter (2-D, `>= min_bytes`) in the decoder layers
  is frozen, trainable ones are **never streamed**: they stay resident, and a warning names the count and GB. If none
  is frozen (an unfrozen model), the selection is unchanged, with a warning that an optimizer cannot step the
  streamed trainable tensors. A frozen model is unchanged and silent.
- **A trainable parameter that offload moves onto the device is moved in place** (`t.data = ...`). An optimizer
  built before `enable_dense_offload` keeps stepping it. A re-wrapped `Parameter` had left it stepping a stale CPU
  copy, so training silently did nothing. Frozen tensors are re-wrapped as before.
- **Freeze for inference:** `model.requires_grad_(False)` before `enable_dense_offload`, then add adapters. README and
  `docs/CHOOSING.md` now say so beside the API.
- **Tests** (`tests/test_dense_offload_trainable.py`) cover:
  - the frozen and unfrozen selections, both identical to before;
  - the kept-resident cases, including a partial fine-tune;
  - the warnings, through `warnings` and through `log=`;
  - AdamW through the offload matching no offload bit for bit, on CPU and CUDA.
