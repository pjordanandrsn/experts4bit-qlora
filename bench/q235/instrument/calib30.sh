#!/bin/bash
# Q235 Amendment 1 calibration on Qwen3-30B-A3B: correctness only, on one CUDA card and its host CPU.
# usage: calib30.sh WORKDIR CHECKPOINT_DIR PYTHON   (this repository on PYTHONPATH; the card claimed for the run)
set -u
B=${1:?work directory}; M=${2:?Qwen3-30B-A3B checkpoint directory}; PY=${3:-python}
I=$(cd "$(dirname "$0")" && pwd); R=$(cd "$I/../../.." && pwd)
export PYTHONPATH=$R TOKENIZERS_PARALLELISM=false
unset E4B_TRAIN_FUSE_QKV E4B_CHUNKED_LM_LOSS GNF4_TRAIN_GEMM
log() { echo "[$(date -u +%FT%TZ)] $*"; }
mkdir -p "$B"
$PY -c "import torch, transformers, bitsandbytes, importlib.metadata as m; print('torch', torch.__version__, 'tf', transformers.__version__, 'bnb', bitsandbytes.__version__, 'gnf4', m.version('grouped-nf4-gemm'), torch.cuda.get_device_name(0))"
log "data"
[ -f "$B/ds_alpaca.json" ] || $PY "$R/bench/tp4/tp4_alpaca.py" --out "$B/ds_alpaca.json" || exit 2
$PY "$I/q235_tokens.py" "$B/ds_alpaca.json" "$M" --rows 8 --seq 2048 --out "$B/tokens.pt" || exit 3
log "1 truncated resident vs streamed (6 layers, cuda)"
$PY "$I/layer_stream_ref.py" "$M" "$B/tokens.pt" --device cuda --layers 6 --resident --out "$B/ref6_resident_cuda" || exit 4
$PY "$I/layer_stream_ref.py" "$M" "$B/tokens.pt" --device cuda --layers 6 --out "$B/ref6_stream_cuda" || exit 5
log "2 the same 6 layers on the host CPU (6 threads, nice 19)"
nice -n 19 $PY "$I/layer_stream_ref.py" "$M" "$B/tokens.pt" --device cpu --threads 6 --layers 6 --resident --out "$B/ref6_resident_cpu" || exit 6
log "3 full depth, streamed bf16 (cuda)"
$PY "$I/layer_stream_ref.py" "$M" "$B/tokens.pt" --device cuda --out "$B/ref48_stream_cuda" || exit 7
log "4 NF4 side, two fresh processes"
$PY "$I/nf4_side.py" "$M" "$B/tokens.pt" --out "$B/nf4_run1" || exit 8
$PY "$I/nf4_side.py" "$M" "$B/tokens.pt" --out "$B/nf4_run2" || exit 9
log "5 compares"
$PY "$I/q235_compare.py" "$B/ref6_stream_cuda" "$B/ref6_resident_cuda" "$B/tokens.pt" --out "$B/cmp_streaming_mechanics.json"
$PY "$I/q235_compare.py" "$B/ref6_resident_cpu" "$B/ref6_resident_cuda" "$B/tokens.pt" --out "$B/cmp_cpu_vs_cuda.json"
$PY "$I/q235_compare.py" "$B/nf4_run1" "$B/ref48_stream_cuda" "$B/tokens.pt" --out "$B/cmp_nf4_vs_bf16.json"
$PY "$I/q235_compare.py" "$B/nf4_run2" "$B/nf4_run1" "$B/tokens.pt" --out "$B/cmp_nf4_fresh_repeat.json"
log "done"
