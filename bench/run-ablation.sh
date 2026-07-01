#!/usr/bin/env bash
# Resumable LoRA-placement ablation: 4 controlled runs (same r=8, steps, lr, data, eval),
# vary only WHAT trains. Each config writes done.flag on success; re-launching skips finished
# configs. Writes to work/ablation/<config>/ (no overwrites of r8/r16 adapters).
#
# HARDENED 2026-06-30 (resume after the /usr/local/cuda-12.4 toolkit was wiped):
#   - CUDA runtime libs come from torch's bundled nvidia/* wheels in .venv-cuda (persistent
#     work volume), NOT the ephemeral /usr/local/cuda-12.4 toolkit -> survives container recycle.
#   - per-config logs go to work/ablation/<config>/run.log (persistent), not /tmp.
set -u
cd /home/node/work/bitsandbytes
. .venv-cuda/bin/activate

# Build LD_LIBRARY_PATH from torch's bundled CUDA libs (cudart/cublas/...). Toolkit-free.
TORCH_NVLIBS=$(python -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(glob.glob(b+'/*/lib'))))")
export LD_LIBRARY_PATH=$TORCH_NVLIBS
export HF_HOME=/home/node/work/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
export STEPS=150 GRAD_ACCUM=4 LR=1e-4 SEQ=256 N_TRAIN=10000 R=8 ALPHA=16 EVAL_EVERY=50 DO_GEN=0

echo "[$(date +%H:%M:%S)] ablation start | LD_LIBRARY_PATH=$TORCH_NVLIBS"
python -c "import torch,bitsandbytes as bnb;assert torch.cuda.is_available();print('[preflight] bnb',bnb.__version__,'on',torch.cuda.get_device_name(0))" || { echo "PREFLIGHT FAILED — aborting"; exit 1; }

run() {  # name experts attn router
  D=/home/node/work/ablation/$1
  if [ -f "$D/done.flag" ]; then echo "SKIP $1 (already done)"; return; fi
  mkdir -p "$D"
  echo "############################## CONFIG: $1 ($(date +%H:%M:%S)) ##############################"
  TRAIN_EXPERTS=$2 TRAIN_ATTENTION=$3 TRAIN_ROUTER=$4 OUT=$D \
    python -u examples/olmoe_experts4bit_qlora.py > "$D/run.log" 2>&1
  rc=$?
  echo "$1 exit=$rc :: $(tr '\r' '\n' < "$D/run.log" | grep -aE 'BEFORE .*-> AFTER' | tail -1)"
  [ $rc -eq 0 ] && touch "$D/done.flag"
}

run experts_only        1 0 0
run attention_only      0 1 0
run experts_attn        1 1 0
run experts_attn_router 1 1 1
echo "[$(date +%H:%M:%S)] ABLATION_ALL_DONE"
