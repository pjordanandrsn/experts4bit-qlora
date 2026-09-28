#!/bin/bash
# Routing-replay cache gate + prefill timing (items 2, 3): main gen run, then the negative control.
set -u
cd /workspace/k3-rel-20260928/v2 || exit 1
PY=/workspace/venv-k3rel/bin/python
export K3_OUT=/workspace/k3-rel-20260928/v2 TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1
echo "[$(date -u +%FT%TZ)] start gen main" >> status.log
K3_MODE=gen K3_NEW_TOKENS=4 K3_PREFILL_REPEAT=1 K3_VERIFY_FREE=1 $PY -u k3_run_rel.py > k3_gen_main.log 2>&1; rc=$?
echo "[$(date -u +%FT%TZ)] end gen main rc=$rc" >> status.log
echo "[$(date -u +%FT%TZ)] start gen control drop_state" >> status.log
K3_MODE=gen K3_NEW_TOKENS=2 K3_CONTROL=drop_state K3_VERIFY_FREE=0 $PY -u k3_run_rel.py > k3_gen_control.log 2>&1; rc=$?
echo "[$(date -u +%FT%TZ)] end gen control rc=$rc" >> status.log
echo "[$(date -u +%FT%TZ)] ALL DONE" >> status.log
