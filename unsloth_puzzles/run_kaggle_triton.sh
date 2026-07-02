#!/usr/bin/env bash
# One-shot Task A (Triton NF4) benchmark for a Kaggle single-T4 notebook cell:
#   !curl -sSL https://raw.githubusercontent.com/pjordanandrsn/experts4bit-qlora/triton-nf4/unsloth_puzzles/run_kaggle_triton.sh | bash
# Fetches the kernel + benchmark, checks it is bit-exact vs bitsandbytes (the benchmark asserts this
# per shape before timing), then times it vs bnb — and vs Unsloth's fast_dequantize if WITH_UNSLOTH=1
# installs cleanly. Prints per-shape median latency + speedup + a geomean. Run on GPU T4 (Internet on).
set -eo pipefail
# Pin to an immutable commit SHA (not the branch) so raw.githubusercontent.com's ~5 min branch cache
# can't serve a stale kernel/benchmark. Bump this SHA when either file changes.
ROOT=https://raw.githubusercontent.com/pjordanandrsn/experts4bit-qlora/fe655b7575fbb36c3fad78bd3604dc2012720383

pip install -q -U bitsandbytes
if [ "${WITH_UNSLOTH:-0}" = "1" ]; then
  # Best-effort: Unsloth's fast_dequantize is the literal rubric baseline, but the package is heavy and
  # may perturb torch/triton — so it is opt-in and non-fatal. The benchmark auto-detects it either way.
  pip install -q unsloth && echo "[unsloth installed]" || echo "[unsloth install failed — reporting ours-vs-bnb only]"
fi

# The kernel is self-contained (torch + triton only); the benchmark imports it from the cwd.
wget -qO triton_nf4.py "$ROOT/experts4bit_qlora/triton_nf4.py"
wget -qO bench_triton_nf4.py "$ROOT/unsloth_puzzles/bench_triton_nf4.py"
python bench_triton_nf4.py
