#!/bin/bash
# P68 rehearsal on the NAS A2000 (NOT the lane's reading): OLMoE-1B-7B-0924 from the LAN model store, P63's baked
# OLMoE arena, this branch's e4b tree, grouped-nf4-gemm main. The size rows are the rehearsal's own: OLMoE-tokenized
# prose from this repository (P64's committed rows are Qwen3 ids). Inside pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel,
# /w = /share/Container/gnf4-interp/p68, /g = /share/Container/gnf4-interp (read-only), /models read-only.
set -u
export PIP_CACHE_DIR=/w/pipcache PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /w/src/e4b
pip install -q --no-input "transformers==5.16.1" "bitsandbytes==0.50.1" accelerate safetensors sentencepiece "huggingface_hub>=0.23" > /w/reh/pip.log 2>&1 || { tail -5 /w/reh/pip.log; exit 9; }
pip install -q --no-input --force-reinstall --no-deps /w/src/gnf4 >> /w/reh/pip.log 2>&1 || exit 9
pip install -q --no-input --no-deps -e /w/src/e4b >> /w/reh/pip.log 2>&1 || exit 9
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader | tee /w/reh/gpu.txt
python -c "import torch, triton, transformers, importlib.metadata as m; print('torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, 'e4b', m.version('experts4bit-qlora'), 'gnf4', m.version('grouped-nf4-gemm'))" | tee -a /w/reh/gpu.txt
MODEL=/models/OLMoE-1B-7B-0924
python - <<'PYR'
import json, hashlib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/models/OLMoE-1B-7B-0924")
text = "\n\n".join(open(p).read() for p in ("bench/p63/P63-PREREG.md", "bench/p64/P64-PREREG.md", "bench/p65/P65-PREREG.md"))
ids = tok(text)["input_ids"]
rows = [ids[i * 512:(i + 1) * 512] for i in range(4)]
assert all(len(r) == 512 for r in rows), [len(r) for r in rows]
json.dump({"prompts": rows, "prompts_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
           "source": "REHEARSAL: OLMoE-tokenized P63/P64/P65 prereg prose"}, open("/w/reh/rows.json", "w"))
print("rows", len(rows), "x", len(rows[0]))
PYR
for S in ${STACKS:-int4 nf4}; do
  echo "=== stack $S $(date -u +%FT%TZ)"
  python -u bench/p68/p68_probe.py --stack $S --model $MODEL --arena /g/p63/work_olmoe/nf4.arena --calib bench/p39/calib.json \
    --size-rows /w/reh/rows.json --out /w/reh/out/$S ${PROBE_ARGS:-} > /w/reh/run_$S.log 2>&1
  echo "stack $S rc=$?"; grep -aE "^P68 |P68ARM|Traceback|Error" /w/reh/run_$S.log | tail -20 | cut -c1-500
done
python bench/p68/p68_reduce.py /w/reh/out --md /w/reh/RESULTS-rehearsal-generated.md --json /w/reh/p68_rep.json > /w/reh/reduce.log 2>&1; echo "reduce rc=$?"
