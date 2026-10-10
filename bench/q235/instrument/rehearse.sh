#!/bin/bash
# Q235 gate 1 on one CUDA card: the box script on a depth-truncated slice of the registered checkpoint (real weights, the box's layer
# shapes), the step and its repeat, both planted errors, and the fidelity read of each against the layer-streamed bf16 reference.
# usage: rehearse.sh WORKDIR SLICE_DIR DS_ALPACA_JSON PYTHON   (this repository on PYTHONPATH; the card claimed for the run)
set -u
B=${1:?work directory}; M=${2:?depth-truncated checkpoint}; DS=${3:?tp4_alpaca.py output}; PY=${4:-python}
I=$(cd "$(dirname "$0")" && pwd); R=$(cd "$I/../../.." && pwd)
export PYTHONPATH=$R TOKENIZERS_PARALLELISM=false
unset E4B_TRAIN_FUSE_QKV E4B_CHUNKED_LM_LOSS GNF4_TRAIN_GEMM E4B_OFFLOAD_STATS
log() { echo "[$(date -u +%FT%TZ)] $*"; }
mkdir -p "$B"
$PY -c "import torch, transformers, bitsandbytes, importlib.metadata as m; print('torch', torch.__version__, 'tf', transformers.__version__, 'bnb', bitsandbytes.__version__, 'gnf4', m.version('grouped-nf4-gemm'), torch.cuda.get_device_name(0))"
log "tokens"; $PY "$I/q235_tokens.py" "$DS" "$M" --rows 8 --seq 2048 --out "$B/tokens.pt" || exit 2
log "reference"; $PY "$I/layer_stream_ref.py" "$M" "$B/tokens.pt" --device cuda --out "$B/ref" || exit 3
log "box"; $PY "$I/q235_box.py" "$M" "$B/tokens.pt" --device cuda --out "$B/box" || exit 4
log "repeat"; $PY "$I/q235_box.py" "$M" "$B/tokens.pt" --device cuda --repeat --out "$B/repeat" || exit 5
for p in routing epilogue; do log "plant $p"; $PY "$I/q235_box.py" "$M" "$B/tokens.pt" --device cuda --plant $p --out "$B/plant_$p" || exit 6; done
log "fidelity reads"
for x in box plant_routing plant_epilogue; do $PY "$I/q235_compare.py" "$B/$x" "$B/ref" "$B/tokens.pt" --out "$B/cmp_$x.json"; done
$PY "$I/q235_compare.py" "$B/repeat" "$B/box" "$B/tokens.pt" --out "$B/cmp_repeat.json"
log "done"
