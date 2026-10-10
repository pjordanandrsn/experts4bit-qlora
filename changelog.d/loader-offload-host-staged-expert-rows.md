### `load_moe_4bit_streaming(offload=True)` on CUDA fuses per-expert rows in host memory

- **What changed.** With `offload=True` on a CUDA device, the streaming loader reads each layer's per-expert rows (Qwen3-MoE,
  OLMoE, the Mixtral family: the checkpoints the MoE conventions parse) into host memory, fuses them there, and moves only
  the fused `gate_up` / `down` stacks to the device, where they are quantized as before. Read straight to the device, a
  layer's rows and its fused stacks coexist there. At Qwen3-235B-A22B shapes that is 4.5 GiB of rows plus 3 GiB per
  intermediate stack: a 12 GB card ran out of memory in layer 0, against the loader's contract that offload holds one
  layer's experts on the device at a time.
- **Same bytes.** Stacking and concatenation are copies, and the quantization still runs on the device from the same input.
  The packed 4-bit bytes and the absmax are bit for bit the same staged and direct; so is the double-quantized absmax that
  `compress_expert_absmax_` derives (a resident load, since that compressor refuses offloaded layers).
- **Unchanged.** The resident (no-offload) path reads to the device exactly as before; the bytes it stores match the previous
  loader's bit for bit. The per-expert arm without a convention (the dedicated-quant families) and checkpoints that already
  store fused stacks are not touched.
- **Cost.** At Qwen3-30B-A3B shapes (a 4-layer slice, warm page cache, RTX A2000) the load took 6.8 s and 6.1 s staged against
  6.2 s and 5.8 s direct. The host peak RSS was about 0.8 GiB higher (10.1 against 9.4 GiB), and the device peak lower
  (1.94 against 2.70 GiB).
- **Test.** `tests/test_loader_host_stage.py`: staged and direct loads of a Qwen3-MoE and a Mixtral checkpoint store the same
  bits, offload on CUDA and resident on any host, with a spy on the fuser that records where its rows were.
