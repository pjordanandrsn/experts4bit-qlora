### SC2g registered (#846): request-level serving of gpt-oss-20b, with e4b's first gpt-oss run through `serve_paged`

- **What runs.** SC2's driver and rule on openai/gpt-oss-20b (`6cee5e81`), one RTX 5090, a new box G in the CUDA 13
  image, four engines, each on its own arithmetic over the checkpoint's MXFP4 experts. Every row carries an arithmetic
  label (sourced) and every e4b ratio is ARITH_MISMATCH:
  - e4b: native MXFP4 decode (`gemv_mxfp4_b32` on int8 activations; K21 on bf16) and NF4 prefill (`E4B_INT4_KEEP_NF4=1`);
  - vLLM: Marlin W4A16 with TRITON_ATTN pinned;
  - SGLang: its gpt-oss defaults, via a new `gptoss` mode in `bench/sc1/sglang/server.sh`;
  - llama.cpp: ggml-org's published MXFP4 GGUF.
- **Design fix from SC2's read.** Both draws repeat ONE Poisson realisation.
- **Predictions.** Q1 TTFT ≥ 2× vLLM's; Q2 TPOT ≤ 1.5×; Q3 vLLM's ceiling ≥ 4 req/s and above e4b's; **Q4 registers
  SC2's post-hoc mechanism** (e4b's stall per interleaved prefill ≥ 10× its per-token cost, R² ≥ 0.9); Q5 every row
  VALID.
- **Files.** `bench/sc2/SC2g-PREREG.md`, `sc2g_box_g.sh`, `sc2g_reduce.py`; `sc2_trace.py` is now staged; grouped-nf4-gemm
  v0.41.0 (e4b 0.48.0's CI pin) is pinned for box G, with `GNF4_TRITON_PREBIND=1` pinned and recorded; `tests/test_sc2g_box.py` executes the child-env, SGLang-engagement and e4b-check paths.
