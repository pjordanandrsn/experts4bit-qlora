#!/usr/bin/env bash
# Offload A/B: two OLMoE runs, identical seed/data/hyperparameters, flipping only OFFLOAD_EXPERTS.
# Proves the CPU-offload of the frozen 4-bit experts is a memory-for-compute trade with NO effect on
# what the adapters learn: the BEFORE eval loss and the AFTER-BEFORE delta must match off-vs-on
# (offload changes tensor location, not math), while peak GPU mem drops (experts leave the card) and
# tokens/s drops (a per-layer PCIe transfer). Resumable like run-ablation.sh; reproduces
# docs/METHODOLOGY.md §11.
#
# Usage:  pip install -e ".[train]"  &&  bash bench/run-offload-ab.sh
#   OUT_DIR   where per-config outputs/logs go   (default ./offload-ab-out)
#   MODEL     HF model id                        (default allenai/OLMoE-1B-7B-0924)
set -u
OUT_DIR=${OUT_DIR:-./offload-ab-out}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Same knobs as the placement ablation, so the offload A/B is directly comparable to METHODOLOGY §7.
export STEPS=${STEPS:-150} GRAD_ACCUM=${GRAD_ACCUM:-4} LR=${LR:-1e-4} SEQ=${SEQ:-256}
export N_TRAIN=${N_TRAIN:-10000} R=${R:-8} ALPHA=${ALPHA:-16} EVAL_EVERY=${EVAL_EVERY:-50} DO_GEN=${DO_GEN:-0}
# Fixed adapter placement across both runs (offload is orthogonal to what trains).
export TRAIN_EXPERTS=${TRAIN_EXPERTS:-1} TRAIN_ATTENTION=${TRAIN_ATTENTION:-1} TRAIN_ROUTER=${TRAIN_ROUTER:-0}

# If bitsandbytes can't find the CUDA runtime, point LD_LIBRARY_PATH at torch's bundled nvidia libs.
# (`nvidia` is a namespace package, so iterate __path__ rather than dirname(__file__), which is None.)
if python -c "import nvidia" 2>/dev/null; then
  export LD_LIBRARY_PATH="$(python -c "import glob,nvidia; print(':'.join(sorted(p+'/lib' for b in nvidia.__path__ for p in glob.glob(b+'/*') if glob.glob(p+'/lib'))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[$(date +%H:%M:%S)] offload A/B start -> $OUT_DIR"
python -c "import torch, bitsandbytes as bnb, experts4bit_qlora; assert torch.cuda.is_available(); print('[preflight] bnb', bnb.__version__, '|', torch.cuda.get_device_name(0))" \
  || { echo "PREFLIGHT FAILED - need '.[train]' installed and a CUDA-enabled torch"; exit 1; }

summarize() {  # name -> pull the comparison numbers out of the run log
  L="$OUT_DIR/$1/run.log"
  echo "  $1: $(tr '\r' '\n' < "$L" | grep -aE 'BEFORE .*-> AFTER' | tail -1)"
  echo "         $(tr '\r' '\n' < "$L" | grep -aE 'peak GPU mem' | tail -1)"
  echo "         last-step: $(tr '\r' '\n' < "$L" | grep -aoE '\([0-9.]+s/step\)' | tail -1)"
}

run() {  # name offload
  D="$OUT_DIR/$1"
  if [ -f "$D/done.flag" ]; then echo "SKIP $1 (already done)"; return; fi
  mkdir -p "$D"
  echo "############################## CONFIG: $1 (offload=$2) ($(date +%H:%M:%S)) ##############################"
  OFFLOAD_EXPERTS=$2 OUT="$D" \
    python -u -m experts4bit_qlora.train > "$D/run.log" 2>&1
  rc=$?
  echo "$1 exit=$rc"
  [ $rc -eq 0 ] && touch "$D/done.flag"
}

run offload_off 0
run offload_on  1

echo "[$(date +%H:%M:%S)] ==== OFFLOAD A/B SUMMARY (expect: identical BEFORE+delta; peak GPU on<off; tok/s on<off) ===="
summarize offload_off
summarize offload_on
echo "[$(date +%H:%M:%S)] OFFLOAD_AB_DONE"
