#!/bin/bash
# P65 rehearsal, CPU step (no GPU): venv over the image torch, e4b (this branch's tree) + P37 pins, gnf4 at the e4b CI
# pin (v0.33.0), and the two calibration texts pre-fetched into /w/hf so the GPU step runs offline.
set -euo pipefail
cd /w
[ -x /w/venv/bin/python ] || python -m venv --system-site-packages /w/venv
/w/venv/bin/pip install -q --cache-dir /w/pipcache /w/e4b "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23"
/w/venv/bin/pip install -q --cache-dir /w/pipcache --force-reinstall --no-deps /w/gnf4
HF_HOME=/w/hf /w/venv/bin/python - <<'PY'
from datasets import load_dataset
a = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
b = load_dataset("allenai/c4", data_files={"v": "en/c4-validation.00001-of-00008.json.gz"}, split="v")
print("texts cached:", len(a), len(b))
PY
/w/venv/bin/python - <<'PY'
import inspect, importlib.metadata as md, torch, triton, transformers, experts4bit_qlora as e
from experts4bit_qlora.engines.int4_experts import calibrate_expert_hessians
assert "activation_means" in inspect.signature(calibrate_expert_hessians).parameters
open("/w/out/versions.txt", "w").write(f"e4b {e.__version__} (branch tree)\ngnf4 {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print(open("/w/out/versions.txt").read())
PY
