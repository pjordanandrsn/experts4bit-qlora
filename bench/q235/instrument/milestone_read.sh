#!/bin/bash
# Q235's milestone read, after the box returns (bench/q235/PREREG-q235.md): the full-depth layer-streamed bf16 reference on the box's own
# tokens (one CUDA card; ~2.4 h on the RTX A2000, its registered slot), then the mechanical verdict, whose exit code this script exits
# with (PASS 0, FAIL 1, VOID 2).
# usage: milestone_read.sh RUN_DIR CHECKPOINT OUT_DIR PYTHON   (this repository on PYTHONPATH; the card claimed for the run)
set -u
RUN=${1:?the box run directory}; M=${2:?the registered checkpoint at ac9c66cc}; OUT=${3:?output directory}; PY=${4:-python}
I=$(cd "$(dirname "$0")" && pwd)
export TOKENIZERS_PARALLELISM=false
log() { echo "[$(date -u +%FT%TZ)] $*"; }
mkdir -p "$OUT/ref"
log "reference (full depth, streamed)"
$PY "$I/layer_stream_ref.py" "$M" "$RUN/receipts/tokens.pt" --device cuda --out "$OUT/ref" || { log "reference failed"; exit 3; }
log "verdict"; $PY "$I/q235_verdict.py" "$RUN" --ref "$OUT/ref" --out "$OUT/verdict.json"; rc=$?
log "done (rc=$rc)"; exit $rc
