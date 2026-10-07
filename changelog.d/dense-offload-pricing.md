### Pricing the dense engines before building them

- **`engines.dense_offload.offload_plan(layers, *, pin, train_prefetch, min_bytes, skip_trainable)`** prices what
  `enable_dense_offload` would do without building it:
  - **Input:** per decoder layer, the `(nbytes, ndim, trainable[, is_param])` of every tensor a handle walks. Mark
    buffers `is_param=False`; it defaults to True.
  - **Selection:** the handle's own rule: 2-D tensors of at least `min_bytes` stream, and trainable ones beside frozen
    parameters stay on the device. A frozen buffer does not keep them, as in `enable_dense_offload`.
  - **Outputs:**
    - the streamed bytes;
    - the pinned host reservation, with each request rounded to a power of two;
    - the staged slots: two layers under `train_prefetch`, one on the synchronous path;
    - what stays on the device;
    - the host-to-device bytes per micro-batch.
  - **Checks:** on Qwen3-32B's NF4 layers it reproduces DQ3's measured host reservation exactly: 15,602,810,880 B
    requested, 17,716,740,096 B reserved. On a toy model on CPU it streams exactly what real handles stream, in each
    freeze mode.
- **`engines.dense_offload.late_bound_4bit_refusal()`** says why offloaded `Linear4bit` training would free no VRAM:
  the installed bitsandbytes differs from the 0.50.2 sources the late-bound backward mirrors. It returns None when
  the backward engages.
- **`engines.chunked_lm_loss.chunked_loss_bytes(supervised_tokens, vocab, hidden=0, chunk=512)`** gives the chunked
  loss's workspace: one chunk's logits at 10 B per logit (`CHUNK_BYTES_PER_LOGIT`, the coefficient the recipe charges
  stock logits), plus the gathered supervised hidden rows.
- **Pure functions:** nothing changes for any caller.
