#!/bin/bash
# P63 rehearsal on the NAS A2000 (NOT the lane's reading): OLMoE-1B-7B-0924 from the LAN model store, this branch's
# e4b tree, grouped-nf4-gemm 66d41c8 (the probed kernels are byte-identical to v0.33.0..main). Inside
# pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel with /w = /share/Container/gnf4-interp/p63, /models read-only.
set -u
export PIP_CACHE_DIR=/w/pipcache PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /w/src/e4b
pip install -q --no-input "transformers==5.16.1" "bitsandbytes==0.50.1" accelerate safetensors sentencepiece "huggingface_hub>=0.23" > /w/logs/reh_pip.log 2>&1 || { tail -5 /w/logs/reh_pip.log; exit 9; }
pip install -q --no-input --force-reinstall --no-deps /w/src/gnf4-66d41c8.tgz >> /w/logs/reh_pip.log 2>&1 || exit 9
pip install -q --no-input --no-deps -e /w/src/e4b >> /w/logs/reh_pip.log 2>&1 || exit 9
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader | tee /w/reh/gpu.txt
python -c "import torch, triton, transformers, importlib.metadata as m; print('torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, 'e4b', m.version('experts4bit-qlora'), 'gnf4', m.version('grouped-nf4-gemm'))" | tee -a /w/reh/gpu.txt
MODEL=/models/OLMoE-1B-7B-0924
if [ ! -e /w/work_olmoe/nf4.arena ]; then
  mkdir -p /w/work_olmoe
  K8_MODEL=$MODEL K8_WORK=/w/work_olmoe python bench/p39/k8_bake.py > /w/reh/bake.log 2>&1 || { tail -5 /w/reh/bake.log; exit 12; }
  rm -rf /w/work_olmoe/nf4snap
fi
for S in ${STACKS:-int4 nf4 int4nf}; do
  echo "=== stack $S $(date -u +%FT%TZ)"
  python -u bench/p63/p63_probe.py --stack $S --model $MODEL --arena /w/work_olmoe/nf4.arena --calib bench/p39/calib.json \
    --out /w/reh/out/$S ${PROBE_ARGS:-} > /w/reh/run_$S.log 2>&1
  echo "stack $S rc=$?"; grep -aE "^P63 |P63ARM|Traceback|Error" /w/reh/run_$S.log | tail -14 | cut -c1-400
done
python bench/p63/p63_reduce.py /w/reh/out --md /w/reh/RESULTS-rehearsal-generated.md --json /w/reh/p63_rep.json > /w/reh/reduce.log 2>&1; echo "reduce rc=$?"
