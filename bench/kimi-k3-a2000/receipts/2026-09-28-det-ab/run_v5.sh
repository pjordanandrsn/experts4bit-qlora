#!/bin/bash
# experts4bit-qlora#761 section 4, the A/B: prefill-only K3 (6 tokens, 1 new token), torch
# deterministic algorithms ON (+ CUBLAS_WORKSPACE_CONFIG) vs OFF, three processes each,
# interleaved so a slow drift in the box cannot line up with one arm. Same driver file,
# same wrapper, same Triton autotune cache (TRITON_CACHE_AUTOTUNING=1, as in v4, and
# TRITON_PRINT_AUTOTUNING=1 so a re-benchmark would show in the log).
set -u
cd /workspace/k3-rel-20260928/v5-det || exit 1
PY=/workspace/venv-k3rel/bin/python
DRV=/workspace/k3-rel-20260928/v3/k3_run_rel.py
export TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 TRITON_CACHE_AUTOTUNING=1 TRITON_PRINT_AUTOTUNING=1
run_one() {   # $1 = run dir, rest = extra env
  local run=$1; shift
  mkdir -p "$run"
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -lt 4700 ]; then
    echo "[$(date -u +%FT%TZ)] SKIP $run: only ${free} MiB free VRAM" >> status.log
    return
  fi
  echo "[$(date -u +%FT%TZ)] start $run ($*) free_vram=${free}MiB" >> status.log
  env "$@" K3_OUT=/workspace/k3-rel-20260928/v5-det/$run K3_MODE=gen K3_NEW_TOKENS=1 \
    K3_VERIFY_CACHE=0 nice -n 10 $PY -u det_wrap.py $DRV > "$run/k3_gen_prefill.log" 2>&1
  echo "[$(date -u +%FT%TZ)] end $run rc=$?" >> status.log
}
for i in 1 2 3; do
  run_one det$i K3_DET=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  run_one plain$i K3_DET=0
done
[ -f phase2.sh ] && . ./phase2.sh
echo "[$(date -u +%FT%TZ)] ALL DONE" >> status.log
