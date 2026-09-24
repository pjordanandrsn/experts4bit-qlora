#!/bin/bash
# P67 NAS env (install only, no GPU): third-party deps into /w/pyenv (pip --target, persists across --rm containers);
# gnf4 at CI's pin into /w/site-ci and at the rehearsal's cut (9206352f, v0.32.1 -- P56's and the standing row's) into
# /w/site-run. torch/triton come from the image (2.8.0+cu128 / 3.4.0), the torch the rented boxes run.
set -euo pipefail
export PIP_CACHE_DIR=/w/pipcache PIP_DISABLE_PIP_VERSION_CHECK=1
python -c "import sys, torch, triton; print('python', sys.version.split()[0], 'torch', torch.__version__, 'triton', triton.__version__)"
mkdir -p /w/pyenv /w/site-ci /w/site-run
pip install -q --no-input --target /w/pyenv "transformers==5.17.0" "safetensors>=0.4" "huggingface_hub>=0.23" sentencepiece tiktoken \
    "datasets>=2.14" "pytest>=7" "fastapi>=0.110" "httpx>=0.27" "jsonschema>=4.18" "ruff==0.15.22" 2>&1 | tail -3
pip install -q --no-input --target /w/pyenv --no-deps "bitsandbytes==0.50.2" "accelerate>=0.30" "peft==0.20.0" 2>&1 | tail -3
rm -rf /w/pyenv/torch /w/pyenv/torch-* /w/pyenv/triton /w/pyenv/triton-* /w/pyenv/nvidia /w/pyenv/numpy /w/pyenv/numpy-* 2>/dev/null || true
pip install -q --no-input --target /w/site-ci --no-deps /w/gnf4-ci 2>&1 | tail -2
pip install -q --no-input --target /w/site-run --no-deps /w/gnf4-run 2>&1 | tail -2
for s in ci run; do
  PYTHONPATH=/w/site-$s:/w/e4b:/w/pyenv python -c "import importlib.metadata as m, transformers, bitsandbytes, experts4bit_qlora as e, nf4_grouped; print('$s: e4b', e.__version__, e.__file__, '| gnf4', m.version('grouped-nf4-gemm'), '| transformers', transformers.__version__, '| bnb', m.version('bitsandbytes'))"
done
