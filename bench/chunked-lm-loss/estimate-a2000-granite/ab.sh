#!/bin/bash
# A/B of granite-3.1-3b-a800m training at T = 6144 (3072 x 2), memory only: S6 stock loss (E4B_CHUNKED_LM_LOSS=0),
# A6 the shipped default (auto: 6144 x 49155 x 4 B = 1.13 GiB of stock fp32 logits >= the 1 GiB gate, so it chunks).
set -u
R=/workspace/chunk-attr
cd "$R"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 E4B_ABSMAX_DQ=0
run() { name=$1; shift; env "$@" "$R/venv/bin/loggetta" train ibm-granite/granite-3.1-3b-a800m-instruct --seq 3072 \
  --micro-batch 2 --steps 2 --experts device --fix expert_kernel=grouped_nf4 --out "$R/runs/$name" > "$R/$name.log" 2>&1
  echo "$name rc=$?"; }
run S6 E4B_CHUNKED_LM_LOSS=0
run A6 E4B_CHUNKED_LM_LOSS=auto
for d in S6 A6; do f=$(ls "$R/runs/$d"/*.json 2>/dev/null | head -1); [ -n "$f" ] && "$R/venv/bin/python" -c "
import json,sys; r=json.load(open(sys.argv[1])); m=r['measured']; print(sys.argv[2], r['status'], 'alloc', round(m['device_peak_bytes']/2**20,1), 'reserved', round(m['device_reserved_peak_bytes']/2**20,1), 'driver', round(m.get('driver_process_peak_bytes',0)/2**20,1))" "$f" "$d"; done
