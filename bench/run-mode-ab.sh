#!/usr/bin/env bash
# Storage-mode validation grid: the same OLMoE QLoRA run and decode measurement across every
# ExpertsNbit scheme, flipping only QUANT_TYPE (+ an OFFLOAD_EXPERTS leg per scheme). Turns the
# README support matrix's non-nf4 rows from correctness-tested into measured: per mode — training
# peak GPU and s/step (resident and offloaded), held-out eval BEFORE -> AFTER (does a
# coarser/finer base change what the adapters learn?), and resident greedy-decode tok/s + peak.
# A validation grid, not a benchmark: the numbers characterize the storage contract on this host.
#
# Hyperparameters match bench/run-offload-ab.sh / METHODOLOGY §7, so nf4 doubles as the in-run
# control AND stays comparable to the ablation. Resumable per leg (done.flag), tolerant of leg
# failures (a 12 GB card OOMs the bf16/fp16 resident legs — they need >=24 GB; the offload legs of
# those modes still run).
#
# Usage:  pip install -e ".[train]"  &&  bash bench/run-mode-ab.sh
#   OUT_DIR   where per-leg outputs/logs go   (default ./mode-ab-out)
#   MODES     space-separated scheme list     (default "nf4 fp4 int8 fp8 bf16 fp16")
#   MODEL     HF model id                     (default allenai/OLMoE-1B-7B-0924)
set -u
OUT_DIR=${OUT_DIR:-./mode-ab-out}
MODES=${MODES:-"nf4 fp4 int8 fp8 bf16 fp16"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Same knobs as the placement ablation / offload A/B, so all three grids are directly comparable.
export STEPS=${STEPS:-150} GRAD_ACCUM=${GRAD_ACCUM:-4} LR=${LR:-1e-4} SEQ=${SEQ:-256}
export N_TRAIN=${N_TRAIN:-10000} R=${R:-8} ALPHA=${ALPHA:-16} EVAL_EVERY=${EVAL_EVERY:-50} DO_GEN=${DO_GEN:-0}
export TRAIN_EXPERTS=${TRAIN_EXPERTS:-1} TRAIN_ATTENTION=${TRAIN_ATTENTION:-1} TRAIN_ROUTER=${TRAIN_ROUTER:-0}
BENCH_TOKENS_N=${BENCH_TOKENS_N:-128}

# If bitsandbytes can't find the CUDA runtime, point LD_LIBRARY_PATH at torch's bundled nvidia libs.
if python -c "import nvidia" 2>/dev/null; then
  export LD_LIBRARY_PATH="$(python -c "import glob,nvidia; print(':'.join(sorted(p+'/lib' for b in nvidia.__path__ for p in glob.glob(b+'/*') if glob.glob(p+'/lib'))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[$(date +%H:%M:%S)] mode A/B start -> $OUT_DIR (modes: $MODES)"
python -c "import torch, bitsandbytes as bnb, experts4bit_qlora; assert torch.cuda.is_available(); print('[preflight] bnb', bnb.__version__, '| torch', torch.__version__, '|', torch.cuda.get_device_name(0))" \
  || { echo "PREFLIGHT FAILED - need '.[train]' installed and a CUDA-enabled torch"; exit 1; }

run_train() {  # name quant_type offload
  D="$OUT_DIR/$1"
  if [ -f "$D/done.flag" ]; then echo "SKIP $1 (already done)"; return; fi
  mkdir -p "$D"
  echo "########## TRAIN $1 (quant_type=$2 offload=$3) ($(date +%H:%M:%S)) ##########"
  QUANT_TYPE=$2 OFFLOAD_EXPERTS=$3 OUT="$D" \
    python -u -m experts4bit_qlora.train > "$D/run.log" 2>&1
  rc=$?
  echo "$1 exit=$rc"
  [ $rc -eq 0 ] && touch "$D/done.flag"
}

run_decode() {  # name quant_type
  D="$OUT_DIR/$1"
  if [ -f "$D/done.flag" ]; then echo "SKIP $1 (already done)"; return; fi
  mkdir -p "$D"
  echo "########## DECODE $1 (quant_type=$2, resident) ($(date +%H:%M:%S)) ##########"
  QUANT_TYPE=$2 BENCH_TOKENS=$BENCH_TOKENS_N OFFLOAD_EXPERTS=0 \
    python -u -m experts4bit_qlora.infer > "$D/run.log" 2>&1
  rc=$?
  echo "$1 exit=$rc"
  [ $rc -eq 0 ] && touch "$D/done.flag"
}

for m in $MODES; do
  run_train  "train_resident_$m" "$m" 0
  run_train  "train_offload_$m"  "$m" 1
  run_decode "decode_$m"         "$m"
done

summarize() {  # name -> the comparison numbers, or the failure reason
  L="$OUT_DIR/$1/run.log"
  [ -f "$L" ] || { echo "  $1: (no log)"; return; }
  if [ ! -f "$OUT_DIR/$1/done.flag" ]; then
    echo "  $1: FAILED - $(tr '\r' '\n' < "$L" | grep -aE 'Error|error|CUDA out of memory' | head -1)"
    return
  fi
  case "$1" in
    train_*)
      echo "  $1: $(tr '\r' '\n' < "$L" | grep -aE 'BEFORE .*-> AFTER' | tail -1)"
      echo "         $(tr '\r' '\n' < "$L" | grep -aE 'peak GPU mem' | tail -1)"
      echo "         last-step: $(tr '\r' '\n' < "$L" | grep -aoE '\([0-9.]+s/step\)' | tail -1)"
      ;;
    decode_*)
      echo "  $1: $(tr '\r' '\n' < "$L" | grep -aE 'decode: .* tok/s' | tail -1)"
      echo "         $(tr '\r' '\n' < "$L" | grep -aE 'peak GPU' | tail -1)"
      ;;
  esac
}

echo "[$(date +%H:%M:%S)] ==== MODE A/B SUMMARY ===="
for m in $MODES; do
  summarize "train_resident_$m"
  summarize "train_offload_$m"
  summarize "decode_$m"
done
echo "[$(date +%H:%M:%S)] MODE_AB_DONE"
