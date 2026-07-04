#!/usr/bin/env bash
# Decode benchmark grid: timed greedy decode (BENCH_TOKENS tokens) across the inference knobs.
# Measures what each lever buys at decode on one card:
#   offload  OFF vs ON      (experts GPU-resident vs streamed per layer from pinned CPU)
#   prefetch OFF vs ON      (offload only: next layer's H2D on a side stream vs serialized)
#   gemv     OFF vs ON      (per-expert dequantize+linear vs bnb fused 4-bit GEMV)
# The fast-path (E4B_DECODE_FASTPATH) stays on except one control row. Each row re-loads the model
# (config is load-time); results append to $OUT_DIR/results.txt as the BENCH lines infer.py prints.
#
# Usage:  pip install -e ".[train]"  &&  bash bench/run-decode-bench.sh
#   OUT_DIR       where logs/results go        (default ./decode-bench-out)
#   MODEL         HF model id                  (default allenai/OLMoE-1B-7B-0924)
#   ADAPTER       adapter .pt to serve         (default: none = base model)
#   BENCH_TOKENS  decode tokens per timed run  (default 128)
set -u
OUT_DIR=${OUT_DIR:-./decode-bench-out}
export MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924}
export ADAPTER=${ADAPTER:-}
export BENCH_TOKENS=${BENCH_TOKENS:-128}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT_DIR"

# If bitsandbytes can't find the CUDA runtime, point LD_LIBRARY_PATH at torch's bundled nvidia libs.
# (`nvidia` is a namespace package, so iterate __path__ rather than dirname(__file__), which is None.)
if python -c "import nvidia" 2>/dev/null; then
  export LD_LIBRARY_PATH="$(python -c "import glob,nvidia; print(':'.join(sorted(p+'/lib' for b in nvidia.__path__ for p in glob.glob(b+'/*') if glob.glob(p+'/lib'))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python -c "import torch, bitsandbytes as bnb, experts4bit_qlora; assert torch.cuda.is_available(); print('[preflight] bnb', bnb.__version__, '|', torch.cuda.get_device_name(0))" \
  || { echo "PREFLIGHT FAILED - need '.[train]' installed and a CUDA-enabled torch"; exit 1; }

run() { # name OFFLOAD_EXPERTS PREFETCH E4B_INFER_GEMV E4B_DECODE_FASTPATH
  local name=$1
  local log="$OUT_DIR/$name.log"
  if [ -f "$OUT_DIR/$name.done" ]; then echo "[skip] $name (done)"; return; fi
  echo "[$(date +%H:%M:%S)] $name ..."
  OFFLOAD_EXPERTS=$2 PREFETCH=$3 E4B_INFER_GEMV=$4 E4B_DECODE_FASTPATH=$5 \
    python -m experts4bit_qlora.infer >"$log" 2>&1 \
    && { grep '^BENCH' "$log" | sed "s/^/$name /" >>"$OUT_DIR/results.txt"; touch "$OUT_DIR/$name.done"; } \
    || echo "[FAIL] $name (see $log)"
}

#    name                  offload prefetch gemv fastpath
run  resident-dequant      0       0        0    1
run  resident-gemv         0       0        1    1
run  resident-maskpath     0       0        1    0     # fast-path control
run  offload-serial        1       0        1    1
run  offload-prefetch      1       1        1    1
run  offload-prefetch-noge 1       1        0    1     # isolates gemv vs prefetch contribution

echo; echo "== results =="; cat "$OUT_DIR/results.txt" 2>/dev/null || echo "(none)"
