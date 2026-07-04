#!/usr/bin/env bash
# Decode grid for a big fused-MoE whose 4-bit experts EXCEED the card (Qwen3-30B-A3B ~15 GB,
# Gemma-4-26B-A4B ~13 GB on a 12 GB A2000). Unlike run-decode-bench.sh (OLMoE, fits resident),
# the resident configs here are EXPECTED to OOM — that is the §11c capability point — so this
# records the OOM instead of faking a number, and focuses on the offload x prefetch x gemv rows.
# Base model (no adapter): the offload/prefetch mechanics and peak GB don't depend on adapter
# presence, and a trained big-model adapter isn't on hand.
#
# Usage:  MODEL=google/gemma-4-26B-A4B bash bench/run-bigmoe-decode.sh
#   OUT_DIR       results/logs dir              (default ./bigmoe-decode-out)
#   MODEL         HF model id                   (required)
#   BENCH_TOKENS  decode tokens per timed run   (default 96)
set -u
: "${MODEL:?set MODEL to a fused-MoE id}"
OUT_DIR=${OUT_DIR:-./bigmoe-decode-out}
export MODEL BENCH_TOKENS=${BENCH_TOKENS:-96} ADAPTER=""
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Models live in the DEFAULT HF cache (~/.cache/huggingface), not work/hf-cache — do not override.
unset HF_HOME
mkdir -p "$OUT_DIR"

# bnb CUDA runtime via torch's bundled nvidia libs (namespace-package __path__ walk).
if python -c "import nvidia" 2>/dev/null; then
  export LD_LIBRARY_PATH="$(python -c "import glob,nvidia; print(':'.join(sorted(p+'/lib' for b in nvidia.__path__ for p in glob.glob(b+'/*') if glob.glob(p+'/lib'))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python -c "import torch, bitsandbytes, experts4bit_qlora; assert torch.cuda.is_available()" \
  || { echo "PREFLIGHT FAILED"; exit 1; }

run() { # name OFFLOAD PREFETCH GEMV
  local name=$1
  local log="$OUT_DIR/$name.log"
  [ -f "$OUT_DIR/$name.done" ] && { echo "[skip] $name"; return; }
  echo "[$(date +%H:%M:%S)] $name ..."
  if OFFLOAD_EXPERTS=$2 PREFETCH=$3 E4B_INFER_GEMV=$4 python -m experts4bit_qlora.infer >"$log" 2>&1; then
    grep '^BENCH' "$log" | sed "s/^/$name /" >>"$OUT_DIR/results.txt"; touch "$OUT_DIR/$name.done"
  elif grep -qiE "out of memory|CUDA out of memory|OutOfMemoryError" "$log"; then
    # Expected for resident configs on a card the experts exceed — record, don't fail the grid.
    echo "$name BENCH offload=$2 prefetch=$3 gemv=$4 RESULT=OOM" >>"$OUT_DIR/results.txt"; touch "$OUT_DIR/$name.done"
    echo "  -> OOM (expected for resident on a card the experts exceed)"
  else
    echo "[FAIL non-OOM] $name (see $log)"
  fi
}

#    name               offload prefetch gemv
run  resident            0       0        1     # expected OOM (experts > VRAM)
run  offload-serial      1       0        1
run  offload-prefetch    1       1        1
run  offload-prefetch-dq 1       1        0

echo; echo "== $MODEL =="; cat "$OUT_DIR/results.txt" 2>/dev/null
touch "$OUT_DIR/all-done.flag"
