#!/bin/bash
# tp1's arm driver (bench/train-parity-20260905/tp1/logs/tp1_train_smoke.py, unchanged) on this box: reference + fused for one
# family, the registered clinical fixture (sha-verified), N=60, seq 512, r=8. Informational: the A2000 is outside tp1's
# train-anchor band, so these rows are parity readings in tp1's units, not licences.
# usage: tp1_pair.sh FAM MODEL OFFLOAD [arms]
FAM=$1; MID=$2; OFF=$3; ARMS=${4:-"reference fused"}
R=/home/node/work/e4b-moegen; OUT=$R/bench/moegen/tp1-a2000; mkdir -p $OUT/logs
DATA=/home/node/work/moegen-tp1/data/ds_clinical.json; SHA=76fb9036de80f3bb495fe4c8894159fcb1d399d2437293e012e264d81949f791
for ARM in $ARMS; do
  echo "[$(date -u +%FT%TZ)] arm $FAM/$ARM offload=$OFF" | tee -a $OUT/summary.txt
  /home/node/work/moegen-venv/bin/python $R/bench/train-parity-20260905/tp1/logs/tp1_train_smoke.py --model $MID --fam $FAM \
    --arm $ARM --steps ${TP1_STEPS:-60} --seq 512 --offload $OFF --data $DATA --data-sha $SHA --out $OUT > $OUT/logs/run_${FAM}_${ARM}.log 2>&1
  echo "[$(date -u +%FT%TZ)] $FAM/$ARM rc=$?" | tee -a $OUT/summary.txt
done
