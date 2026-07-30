#!/bin/bash
# The second model's ten cells: 5 datasets x 2 arms, 200 steps, per
# docs/PREREG-flagship-matrix-model2.md. Receipts land per cell, so a pod death
# mid-matrix keeps everything that finished.
set -u
cd /workspace/n17 || exit 1
LOG=/workspace/n17/matrix.log
exec > >(tee -a "$LOG") 2>&1

FAILED=0
[ -f /workspace/n17/SETUP_OK ] || { echo "REFUSING: no SETUP_OK sentinel -- gates never passed"; exit 1; }

PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c "import torch" >/dev/null 2>&1; then PY="$cand"; break; fi
done
# n17_setup.sh guards this and the matrix did not. With PY empty, `$PY -u
# n17_cell.py` runs `-u` as a command: every cell dies "command not found" and
# the matrix reports ten opaque failures instead of one clear fatal line.
[ -n "$PY" ] || { echo "FATAL: no interpreter has torch — refusing to start"; exit 1; }

MODEL="${MODEL:-google/gemma-4-26B-A4B}"
STEPS="${STEPS:-200}"
SHORT="$(basename "$MODEL")"
mkdir -p /workspace/n17/cells
echo "=== n17 matrix $(date -u +%FT%TZ) model=$MODEL steps=$STEPS ==="

for ds in clinical code finance legal support; do
  for arm in reference fused; do
    OUT="/workspace/n17/cells/${SHORT}__${ds}__${arm}.json"
    if [ -s "$OUT" ]; then echo "=== SKIP ${ds}/${arm} (receipt exists) ==="; continue; fi
    echo "=== CELL ${MODEL} / ${ds} / ${arm}  ($(date -u +%H:%M:%S)) ==="
    $PY -u /workspace/n17/n17_cell.py --model "$MODEL" --dataset "$ds" \
        --arm "$arm" --steps "$STEPS" --out "$OUT"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "CELL FAILED rc=$rc ${ds}/${arm} -- continuing so the rest of the matrix still runs"
      FAILED=$((FAILED + 1))
      echo "${ds}/${arm} rc=$rc" >> /workspace/n17/FAILED_CELLS
    fi
    $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
  done
done

# A "done" sentinel that fires after failures is the same defect as
# `echo "$(date) exit=$?"`: a completion signal that cannot express failure, which
# downstream automation reads as success. Count the receipts that actually exist
# and say which state this is.
N_OK=$(ls /workspace/n17/cells/*.json 2>/dev/null | wc -l | tr -d " ")
echo "MATRIX DONE $(date -u +%FT%TZ) — receipts=${N_OK}/10 failed_cells=${FAILED}"
ls -la /workspace/n17/cells/
if [ "$N_OK" = "10" ] && [ "$FAILED" = "0" ]; then
  echo "N17_ALL_DONE receipts=10 failed=0" > /workspace/n17/N17_ALL_DONE
  exit 0
fi
echo "N17_PARTIAL receipts=${N_OK}/10 failed=${FAILED}" > /workspace/n17/N17_PARTIAL
echo "MATRIX INCOMPLETE — ${N_OK}/10 receipts, ${FAILED} failed cell(s)"
exit 1
