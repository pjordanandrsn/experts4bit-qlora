#!/bin/bash
# Guard (a), CPU only (no GPU claim needed): run inside gpu-dev after gate 1 ends. Uses the calibration's venv and tokens.
set -u
B=/workspace/prof/fam235; C=/workspace/prof/q235cal; M=/models/Qwen3-30B-A3B; PY=/workspace/venv/bin/python
cd $B && rm -rf e4b && mkdir e4b && tar xzf e4b-060d5f2d.tgz -C e4b
export PYTHONPATH=$B/e4b HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[$(date -u +%FT%TZ)] guard_a start"
nice -n 19 $PY $B/guard_a.py $B/e4b $M $C $B/guard_a.json; rc=$?
echo "[$(date -u +%FT%TZ)] guard_a done rc=$rc"; exit $rc
