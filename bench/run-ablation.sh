#!/usr/bin/env bash
# Resumable LoRA-placement ablation: 4 controlled runs (same r, steps, lr, data, eval),
# varying only WHAT trains (experts / attention / router). Each config writes done.flag on
# success; re-launching skips finished configs. Reproduces docs/METHODOLOGY.md §1-§8.
#
# Usage:  pip install -e ".[train]"  &&  bash bench/run-ablation.sh
#   OUT_DIR   where per-config outputs/logs go   (default ./ablation-out)
#   MODEL     HF model id                        (default allenai/OLMoE-1B-7B-0924)
set -u
OUT_DIR=${OUT_DIR:-./ablation-out}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export STEPS=${STEPS:-150} GRAD_ACCUM=${GRAD_ACCUM:-4} LR=${LR:-1e-4} SEQ=${SEQ:-256}
export N_TRAIN=${N_TRAIN:-10000} R=${R:-8} ALPHA=${ALPHA:-16} EVAL_EVERY=${EVAL_EVERY:-50} DO_GEN=${DO_GEN:-0}

# If bitsandbytes can't find the CUDA runtime, point LD_LIBRARY_PATH at torch's bundled nvidia libs.
# (`nvidia` is a namespace package, so iterate __path__ rather than dirname(__file__), which is None.)
if python -c "import nvidia" 2>/dev/null; then
  export LD_LIBRARY_PATH="$(python -c "import glob,nvidia; print(':'.join(sorted(p+'/lib' for b in nvidia.__path__ for p in glob.glob(b+'/*') if glob.glob(p+'/lib'))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[$(date +%H:%M:%S)] ablation start -> $OUT_DIR"
python -c "import torch, bitsandbytes as bnb, experts4bit_qlora; assert torch.cuda.is_available(); print('[preflight] bnb', bnb.__version__, '|', torch.cuda.get_device_name(0))" \
  || { echo "PREFLIGHT FAILED - need '.[train]' installed and a CUDA-enabled torch"; exit 1; }

run() {  # name experts attn router
  D="$OUT_DIR/$1"
  if [ -f "$D/done.flag" ]; then echo "SKIP $1 (already done)"; return; fi
  mkdir -p "$D"
  echo "############################## CONFIG: $1 ($(date +%H:%M:%S)) ##############################"
  TRAIN_EXPERTS=$2 TRAIN_ATTENTION=$3 TRAIN_ROUTER=$4 OUT="$D" \
    python -u -m experts4bit_qlora.train > "$D/run.log" 2>&1
  rc=$?
  echo "$1 exit=$rc :: $(tr '\r' '\n' < "$D/run.log" | grep -aE 'BEFORE .*-> AFTER' | tail -1)"
  [ $rc -eq 0 ] && touch "$D/done.flag"
}

run experts_only        1 0 0
run attention_only      0 1 0
run experts_attn        1 1 0
run experts_attn_router 1 1 1
echo "[$(date +%H:%M:%S)] ABLATION_ALL_DONE"
