### The routed-expert combine runs over row chunks on large rows (same bytes)

- The training combine (`_ScatterCombine`) and the inference combine build a `[tokens*k, hidden]` fp32 image: the forward's scatter and
  weight multiply, the backward's weight gradient (a per-row sum) and down gradient. On Qwen3-30B-A3B's packed 4,096-token rows that is
  256 MiB per copy. Each transient held about three copies: 768 MiB above its inputs in forward and in backward, measured on an RTX A2000.
- Every one of those operations is row-wise. Once the whole image would reach 128 MiB (`COMBINE_CHUNK_MIN_BYTES`), they now run over row
  chunks of about 32 MiB (`COMBINE_CHUNK_BYTES`: 4,096 rows at hidden 2048), writing into preallocated gradients. The forward and both
  gradients are `torch.equal` to the whole-tensor path on CPU and on the A2000, at 1,024- to 32,768-row chunks. At 4,096-row chunks the
  forward's transient falls from 768 to 320 MiB and the backward's from 768 to 256 MiB
  (`bench/combine-chunk/receipts/combine_chunk_a2000.json`). `tests/test_moe_keep.py`'s packed-shape CUDA test reads the backward
  at 0.269 GB, against 0.805 GB after #1296, with both gradients still `torch.equal` to the original composite.
- Every chunk keeps at least 16 rows, and a short tail joins the chunk before it. Below 16 rows, ATen's CUDA reduction picks its
  per-row thread layout from the row count, so the weight gradient's summation order can change: a 7-row chunk at width 128 did, on
  the A2000. From 16 rows up, every row gets the same layout on any GPU, for widths from 128 to 32,767.
- Under the gate the whole-tensor path runs as before; at TC1's field recipe a call's image is about 70 MiB. `E4B_COMBINE_CHUNK=0` turns
  chunking off. `COMBINE_STATS` counts chunked forwards and backwards.
- What it does to a training step's peak and speed is for a TC1 box to read.
