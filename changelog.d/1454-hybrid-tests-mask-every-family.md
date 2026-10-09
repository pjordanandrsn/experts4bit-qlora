### Tests: every CPU hybrid builder imports its modeling module with the CUDA-only kernels masked (#1454 follow-up)

- **What changed.** `tests/hybrid_reference.py` masks `mamba_ssm` too. `reference_modeling(family)` now takes any
  family, and `bound_kernels` reads a binding from the fallback closure, not from a function name. The builders in
  `test_moe_keep.py`, `test_chunked_lm_loss.py`, `test_moegen_structural.py` and `test_fusions_default.py` import
  through it. `test_moe_keep.py` no longer skips a hybrid that fails to run.
- **Why.** Where `causal_conv1d` is installed, ten cases failed or skipped. `test_chunked_lm_loss.py` failed its two
  tests on each of `granitemoehybrid`, `lfm2_moe`, `qwen3_5_moe` and `nemotron_h`, and `test_moe_keep.py` skipped
  `lfm2_moe` and `qwen3_5_moe` with `Expected x.is_cuda()`. CI installs none of these packages.
- **Who is affected.** Contributors running the suite where `causal_conv1d`, `fla` or `mamba_ssm` is installed.
  Package code is unchanged.
- **Tests (`tests/test_hybrid_reference.py`):**
  - In a fresh process, stand-ins for all three packages shadow whatever is installed. On Qwen3.5 and
    GraniteMoeHybrid, a plain import binds every stand-in function and a CPU forward raises. `reference_modeling`
    refuses the module and names every bound function. On Qwen3.5-MoE, LFM2-MoE and Nemotron-H, the helper's import
    binds none and the forward runs.
  - No test module imports a kernel-binding modeling module, or one of its model classes, at collection. The family
    list is read from the installed transformers' source.
