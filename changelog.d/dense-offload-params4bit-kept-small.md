### `enable_dense_offload` no longer raises on 4-bit projections under `MIN_BYTES`

- **What changed.** A frozen parameter under `MIN_BYTES` (1 MiB) stays resident and is moved to the offload device when it
  sits elsewhere. That branch re-wrapped the moved tensor in `torch.nn.Parameter`, which refuses a subclass whose `detach()`
  returns a plain tensor -- bitsandbytes' `Params4bit` does -- so any model whose packed 4-bit projections fall under 1 MiB
  raised at `enable_dense_offload` when its layers were staged on another device (the host, for instance). Such a subclass
  is now stored as `.to()` returns it, with its quantization state (quantizing on the move when it was not yet); a plain
  parameter is re-wrapped as before.
- **Where it reproduced.** torch 2.11.0, 2.12.1 and 2.14.1 alike, with bitsandbytes 0.50.2: not a torch-version change.
- **Test.** `tests/test_dense_offload_param_subclass.py`. CPU (in CI) reaches the branch by offloading onto `cpu:0`; CUDA
  offloads from the host. Both cover packed and quantize-on-move weights, and the forward is bit for bit the same model
  moved with `Module.to`. Run on torch 2.11.0+cu128 (RTX A2000, CPU and CUDA) and torch 2.14.1 (CPU).
