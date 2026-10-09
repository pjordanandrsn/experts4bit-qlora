### Tests: CPU hybrid tests import transformers' Gated DeltaNet modules with the CUDA-only kernels masked (fix; #1454)

- **What changed.** `tests/hybrid_reference.py` (`reference_modeling`) imports the Qwen3.5-MoE, Qwen3.5 and
  Qwen3-Next modeling modules with `causal_conv1d` and `fla` masked. The first import then binds transformers' torch
  functions. If a family was already imported with a kernel bound, the helper refuses it and names the functions.
  - `tests/test_linear_state.py` no longer imports the modeling module at module scope. It gets the module from the
    helper and keeps a module-level `importorskip` on the configuration module only.
  - The fixtures in `test_fam_shape_order.py`, `test_fam_speed.py` and `test_fam_box.py` now use the helper instead
    of their own masks.
- **Why.** transformers binds those kernels when a modeling module is first imported. Both are CUDA-only. The
  collection-time import in `test_linear_state.py` came before any mask. On a machine with `causal_conv1d` installed,
  every CPU hybrid test in the session then failed with `Expected x.is_cuda() to be true` (14 tests across these four
  files). CI installs neither package, so CI stayed green.
- **Who is affected.** Contributors running the suite where either package is installed. Package code is unchanged.
- **Tests (`tests/test_hybrid_reference.py`):**
  - In a fresh process, a stand-in `causal_conv1d` shadows whatever is installed. A plain import binds it and a CPU
    forward raises from it. `reference_modeling` refuses that module, names both functions, and runs the forward on a
    family it imports itself.
  - No test module imports one of these modeling modules, or a model class that lazily imports one, at collection.
