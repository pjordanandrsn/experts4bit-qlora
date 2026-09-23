#!/bin/bash
# P65 rehearsal, GPU step for one model: bake the NF4 arena (P39's k8_bake.py), run the census. NOT a reading.
set -uo pipefail
FAM=$1; MODEL=$2; NSEQ=$3; LAYERS=${4:-}
export HF_HOME=/w/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export E4B_INT4_HESSIAN_BUDGET_GB=${BUDGET:-2}
PY=/w/venv/bin/python
mkdir -p /w/out /w/logs
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,power.limit --format=csv,noheader | tee /w/out/forensics_$FAM.txt
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /w/logs/vram_$FAM.log 2>/dev/null; sleep 5; done ) & MON=$!
t0=$(date +%s)
if [ ! -e /w/work_$FAM/nf4.arena ]; then
  mkdir -p /w/work_$FAM
  K8_MODEL=$MODEL K8_WORK=/w/work_$FAM $PY /w/e4b/bench/p39/k8_bake.py > /w/logs/bake_$FAM.log 2>&1
  grep -a "BAKE" /w/logs/bake_$FAM.log | tail -1
fi
t1=$(date +%s)
E4B_MODEL_ID=$MODEL $PY /w/e4b/bench/p65/p65_census.py --family $FAM --model $MODEL --arena /w/work_$FAM/nf4.arena \
  --calib /w/e4b/bench/p39/calib.json --nseq $NSEQ ${LAYERS:+--layers $LAYERS} ${FIRST:+--first-layers $FIRST} --out /w/out/census_$FAM.json > /w/logs/census_$FAM.log 2>&1
rc=$?
t2=$(date +%s)
kill $MON 2>/dev/null
echo "family=$FAM first=${FIRST:-all} rc=$rc bake_s=$((t1-t0)) census_s=$((t2-t1)) budget_gb=$E4B_INT4_HESSIAN_BUDGET_GB peak_vram_used_mb(all tenants)=$(sort -n /w/logs/vram_$FAM.log | tail -1)" | tee -a /w/out/timing.txt
tail -5 /w/logs/census_$FAM.log
exit $rc
