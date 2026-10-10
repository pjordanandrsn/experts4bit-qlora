### On a ROCm build of torch, the fused FP8 KV append, the CUDA-graph paths and the prebound launcher refuse in words that name AMD

ROCm answers `torch.cuda.get_device_capability` with gfx numbers (MI300X / gfx942 is `(9, 4)`), which pass the
sm_89 rule the fused FP8 KV append and the bucketed decode graphs were gated on, so the default server would have
engaged them on hardware whose fp8 is FNUZ or absent. No AMD card has run this package's suites, so the gates ask the
vendor first (`experts4bit_qlora._rocm`, keyed on `torch.version.hip`):

- `fp8_paged_kv.fused_append_unsupported(capability, vendor=None)` returns an AMD reason (`vendor` is `"nvidia"`,
  `"amd"`, or `None` for the running torch's). The KV cache then keeps the eager append, and an explicit
  `E4B_FUSED_KV_APPEND=1` is refused in words.
- `E4B_PAGED_GRAPHS=auto` decodes eagerly on AMD, and `1` is refused; `PagedModelRunner.enable_decode_graphs`,
  `enable_prefill_graph` (`E4B_PAGED_PREFILL_GRAPH`, which reports the reason on `/health`) and `capture_decode` refuse
  on AMD, and the speculative decoder's post step stays eager.
- The prebound launch path (`E4B_TRITON_PREBIND`) stays off on ROCm even when forced; `prebind_unsupported()` says why.

NVIDIA behaviour is unchanged. `tests/test_rocm_refusals.py` (CPU, mocked ROCm) pairs each refusal with the same call
under an NVIDIA build at the same capability tuple.
