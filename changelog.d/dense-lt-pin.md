### `E4B_DENSE_LT_PIN`: a pinned cuBLASLt GEMM for named dense projections at 2..8 rows, opt-in (serving roadmap lever K, step K1)

`engines/dense_lt_pin.py`, following the design note `bench/dense-small-m/DESIGN.md`. `E4B_DENSE_LT_PIN=o_proj[,router]`
routes the named projections' 2..8-row calls through one cuBLASLt algorithm configuration, pinned per (cuBLASLt version,
compute capability, SM count, (N, K), M). It is off by default, and no speed is claimed.

- **The pin table (`LT_PINS`) is empty:** a registered selection on the reading box fills it by a code PR. Until then
  the knob is refused at build for want of a pin.
- **The library:** ctypes against the one `libcublasLt` mapped in the process, which is torch's, read from
  `/proc/self/maps` and opened `RTLD_NOLOAD`, with its version in the key. No new dependency.
- **Capture and determinism:** one persistent workspace sized from `cublasLtMatmulAlgoCheck`, and one serving stream,
  enforced. Reduction schemes are limited to 0, 2 and 4 (INPLACE refused).
- **Rows:** 2..8 rows with a pin route through it. 2..8 rows without one pass to what ran before and are counted, by
  site, M and prefill/decode, in `/health`'s `dense_lt_pin`. A captured decode bucket without a pin is refused at build.
- **o_proj** wraps an exact bias-free `nn.Linear`, or K35's one-row `DenseGemvLinear`; one row still reaches K35.
- **The router** routes only on the router epilogue's softmax-then-top-k forward. That forward now marks its kind and
  calls `router_linear`, which is `F.linear` exactly when no pin is set.
- **The adapter census:** classes the knob as adapter-neutral. It reads the merged weight and re-quantizes nothing.
- **Tests:**
  - `tests/test_dense_lt_pin.py` (CPU, a fake library, 24): every refusal, the routing, the counts, and the
    serve_paged and epilogue wiring;
  - `tests/test_dense_lt_pin_gpu.py` (GPU, the real library): within 1.5e-2 of `F.linear`, bitwise run to run, bitwise
    under graph replay, and a rejected configuration refused. All 34 passed on an RTX A4000 shakedown.
