#!/bin/bash
# Per-token perplexity pass (item 4), queued behind the v2 gate runs so the A2000 runs one K3 at a time.
set -u
cd /workspace/k3-rel-20260928/v3 || exit 1
i=0; while [ $i -lt 360 ]; do grep -q "ALL DONE" ../v2/status.log 2>/dev/null && break; sleep 10; i=$((i+1)); done
grep -q "ALL DONE" ../v2/status.log || { echo "[$(date -u +%FT%TZ)] v2 never finished -- not starting" >> status.log; exit 1; }
PY=/workspace/venv-k3rel/bin/python
export K3_OUT=/workspace/k3-rel-20260928/v3 TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1
echo "[$(date -u +%FT%TZ)] start ppl per-token" >> status.log
K3_MODE=ppl K3_PPL_TOKENS=128 $PY -u k3_run_rel.py > k3_ppl.log 2>&1; rc=$?
echo "[$(date -u +%FT%TZ)] end ppl rc=$rc" >> status.log
echo "[$(date -u +%FT%TZ)] ALL DONE" >> status.log
