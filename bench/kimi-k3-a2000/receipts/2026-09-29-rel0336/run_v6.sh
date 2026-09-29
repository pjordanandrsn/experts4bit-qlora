#!/bin/bash
# Kimi-K3 on the RELEASED grouped-nf4-gemm 0.33.6 (venv-k3rel upgraded 2026-09-29): three prefill-only
# processes, deterministic mode OFF, no shadow module, through the same trace wrapper as v5-det.
# Expected if the release carries #410: all three identical at every MoE call, and identical to
# v5-det's det and ord runs (p 0.7144126892089844, 6,118 rows).
set -u
cd /workspace/k3-rel-20260928/v6-rel0336 || exit 1
PY=/workspace/venv-k3rel/bin/python
DRV=/workspace/k3-rel-20260928/v3/k3_run_rel.py
export TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 TRITON_CACHE_AUTOTUNING=1 TRITON_PRINT_AUTOTUNING=1
for i in 1 2 3; do
  run=rel$i
  mkdir -p $run
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -lt 4700 ]; then
    now=$(date -u +%FT%TZ); echo "[$now] SKIP $run: only ${free} MiB free VRAM" >> status.log; continue
  fi
  now=$(date -u +%FT%TZ); echo "[$now] start $run free_vram=${free}MiB" >> status.log
  K3_DET=0 K3_OUT=/workspace/k3-rel-20260928/v6-rel0336/$run K3_MODE=gen K3_NEW_TOKENS=1 K3_VERIFY_CACHE=0 \
    nice -n 10 $PY -u det_wrap.py $DRV > $run/k3_gen_prefill.log 2>&1
  rc=$?
  now=$(date -u +%FT%TZ); echo "[$now] end $run rc=$rc" >> status.log
done
now=$(date -u +%FT%TZ); echo "[$now] ALL DONE" >> status.log
