#!/bin/bash
# Allocator replays of granite-3.1-3b-a800m training under experts4bit-qlora's chunked LM loss.
# P1: T = 1024, E4B_CHUNKED_LM_LOSS=1, a plain train run with no allocator recording (does the crash need the recorder?).
# C1: T = 1024 with E4B_CHUNKED_LM_LOSS=1 (two 512-token chunks), replayed.
# C2: T = 6144 with the shipped default (auto: stock fp32 logits 6144 x 49155 x 4 = 1.13 GiB >= the 1 GiB gate), replayed.
set -u
R=/workspace/chunk-attr
cd "$R/loggetta"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 E4B_ABSMAX_DQ=0
{
  echo "start=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  E4B_CHUNKED_LM_LOSS=1 "$R/venv/bin/loggetta" train ibm-granite/granite-3.1-3b-a800m-instruct --seq 512 --micro-batch 2 \
    --steps 3 --experts device --fix expert_kernel=grouped_nf4 --out "$R/runs/P1"; echo "P1 rc=$?"
  rm -rf "$R/runs/C1" "$R/runs/C2"
  E4B_CHUNKED_LM_LOSS=1 "$R/venv/bin/python" bench/train_residual.py "$R/granite-t1024.json" --out "$R/runs/C1"; echo "C1 rc=$?"
  "$R/venv/bin/python" bench/train_residual.py "$R/granite-t6144.json" --out "$R/runs/C2"; echo "C2 rc=$?"
  for d in C1 C2; do test -f "$R/runs/$d/residual.json" && "$R/venv/bin/python" bench/train_residual.py --summarize "$R/runs/$d"; done
  echo "end=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} > "$R/chunk.log" 2>&1
grep -E "rc=|status OK|estimate |SystemError|Error:" "$R/chunk.log" | head -20
