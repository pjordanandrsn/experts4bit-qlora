#!/bin/bash
# Determinism test: two prefill-only processes with Triton's autotune choice persisted to
# disk. Run A autotunes and writes the choice; run B reads it. Identical p(' Paris') and
# expert-row counts across the two processes = the run-to-run drift was autotuning.
set -u
cd /workspace/k3-rel-20260928/v4 || exit 1
i=0; while [ $i -lt 480 ]; do grep -q "ALL DONE" ../v3/status.log 2>/dev/null && break; sleep 10; i=$((i+1)); done
grep -q "ALL DONE" ../v3/status.log || { echo "[$(date -u +%FT%TZ)] v3 never finished -- not starting" >> status.log; exit 1; }
PY=/workspace/venv-k3rel/bin/python
export TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 TRITON_CACHE_AUTOTUNING=1 TRITON_PRINT_AUTOTUNING=1
for run in A B; do
  mkdir -p $run
  echo "[$(date -u +%FT%TZ)] start $run" >> status.log
  K3_OUT=/workspace/k3-rel-20260928/v4/$run K3_MODE=gen K3_NEW_TOKENS=1 K3_VERIFY_CACHE=0 \
    $PY -u ../v3/k3_run_rel.py > $run/k3_gen_prefill.log 2>&1; rc=$?
  echo "[$(date -u +%FT%TZ)] end $run rc=$rc" >> status.log
done
echo "[$(date -u +%FT%TZ)] ALL DONE" >> status.log
