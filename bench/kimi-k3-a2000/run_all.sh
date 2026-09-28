#!/bin/bash
# K3 full depth on the A2000, released stack (e4b 0.37.5 / gnf4 0.33.4): gen, ppl, route.
set -u
cd /workspace/k3-rel-20260928 || exit 1
PY=/workspace/venv-k3rel/bin/python
export K3_OUT=/workspace/k3-rel-20260928 TMPDIR=/workspace/tmp
for spec in "gen 4" "ppl 0" "route 0"; do
  set -- $spec
  echo "[$(date -u +%FT%TZ)] start $1" >> status.log
  K3_MODE=$1 K3_NEW_TOKENS=${2:-4} K3_PPL_TOKENS=128 $PY k3_run_rel.py > k3_$1.log 2>&1; rc=$?
  echo "[$(date -u +%FT%TZ)] end $1 rc=$rc" >> status.log
done
echo "[$(date -u +%FT%TZ)] ALL DONE" >> status.log
