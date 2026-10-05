### Offloaded training through bitsandbytes `Linear4bit` now frees the evicted weights (behaviour change; #1183)

- **The bug.** bnb 0.50.2's `MatMul4Bit` keeps the frozen packed weight as a ctx attribute (`ctx.tensors = (None, B)`)
  whenever the input needs grad. Checkpointing cannot drop it, so every layer's weight stayed alive from forward to
  backward and dense offload saved no VRAM in training. DQ3's streamed arm ran out of memory on a 5090 (#1083).
  This is bnb 0.50.2 behaviour, worked around locally and not reported upstream.
- **The fix.** Every offloaded `Linear4bit`'s grad-mode matmul now goes through `_LateBoundMatMul4Bit`:
  - its forward is bnb's own no-grad `gemm_4bit` call;
  - its backward is `MatMul4Bit.backward`'s expressions on the weight bound at backward time;
  - only the module is kept.

  Gradients are bitwise identical to stock bnb. Inference takes the stock forward.
- **The pin.** The mirror is pinned by sha256 of the four bnb 0.50.2 sources it reproduces. On a mismatch it warns
  and keeps stock bnb.
- **Measured on the A2000** (DQ3 rehearsal, 4 layers at Qwen3-32B width): the streamed arm now peaks at 5.37 GiB and
  synchronous offload at 4.91 GiB, against 5.82 GiB resident. Before the fix, the streamed arm was at 6.47 GiB.
