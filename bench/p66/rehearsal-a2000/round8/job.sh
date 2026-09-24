#!/bin/bash
# P66 rehearsal, round 8 (after the rental-lesson changes: the 8-stream egress probe, output hashes in G3): (a) P66_MODE=prove through the REAL install path at the
# new main SHAs, with the READING's floors in force so the A2000 misses VRAM and RAM and the proof must RECORD them,
# (b) P66_MODE=full with the final staged tree. NOT a reading.
set -u
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/w/hfcache
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$FREE" -ge 8500 ] || { echo "ABORT: only ${FREE} MiB free on the shared A2000"; exit 75; }
E4B=064462036b5821a49a21edbe0ee2ad6776de9ba0; G4=68a125099c54ddbcf5c26195bd480f79ed9b0724
R=/root/p66
stage(){ rm -rf $R/* 2>/dev/null; mkdir -p $R
  cp /w/src/e4b/bench/p66/p66_run.sh /w/src/e4b/bench/p66/p66_census.py /w/src/e4b/bench/p66/p66_reduce.py \
     /w/src/e4b/bench/p66/p66_step.py /w/src/e4b/bench/p66/staged.sha256 /w/src/e4b/bench/p39/k8_bake.py \
     /w/src/e4b/bench/hybrid-g9/step_decomp.py /w/src/e4b/bench/hybrid-g9/f1/step_budget.py $R/; }
BASE="P66_INSTANCE_ID=nas-a2000 E4B_SHA=$E4B GNF4_SHA=$G4 P66_REHEARSAL_CLASS=A2000 P66_REHEARSAL_QWEN_SNAPSHOT=/models/OLMoE-1B-7B-0924"
COMMON="$BASE P66_REHEARSAL_MIN_VRAM_MB=8500 P66_REHEARSAL_MIN_DISK_GB=50 P66_REHEARSAL_MIN_RAM_GB=16 P66_REHEARSAL_PIN_GIB=4"
echo "===== (a) prove, real install, reading floors in force"
stage; cd $R; NONCE=$(python -c 'import secrets; print(secrets.token_hex(32))'); T0=$(date +%s)
env -u PYTHONPATH $BASE P66_RUN_ID=p66-rehearsal-a2000-r8-prove P66_RUN_NONCE=$NONCE P66_DEADLINE_EPOCH=$(( $(date +%s) + 828 )) P66_MODE=prove \
  bash p66_run.sh > outer.log 2>&1
echo "prove rc=$? seconds=$(( $(date +%s) - T0 )) exit=$(cat P66_EXIT_CODE.$NONCE 2>/dev/null) proved=$(ls P66_PROVED.$NONCE 2>/dev/null) success=$(ls P66_SUCCESS.$NONCE 2>/dev/null)"
grep -v "^\s*$" outer.log | tail -25; grep -E "floor_would|PROVE" summary.txt
rm -rf /w/work/r8prove; mkdir -p /w/work/r8prove; cp -R $R/* /w/work/r8prove/; rm -rf /w/work/r8prove/src
pip install -q --no-input --target /w/pydeps --no-deps --upgrade /w/src/e4b > /w/work/r8_pip_e4b.log 2>&1 || echo "PIP e4b into pydeps failed"
echo "===== (b) full, the committed tree"
stage; cd $R; NONCE=$(python -c 'import secrets; print(secrets.token_hex(32))'); T0=$(date +%s)
env $COMMON P66_RUN_ID=p66-rehearsal-a2000-r8-full P66_RUN_NONCE=$NONCE P66_DEADLINE_EPOCH=$(( $(date +%s) + 5400 )) \
  P66_REHEARSAL_SKIP_INSTALL=1 P66_REHEARSAL_GNF4_SRC=/w/src/gnf4 P66_REHEARSAL_NF4_FAMILY=olmoe-1b-7b \
  P66_REHEARSAL_NF4_ARENA=/w/work/olmoe/nf4.arena P66_REHEARSAL_GPTOSS_SNAPSHOT=/models/gpt-oss-20b \
  P66_REHEARSAL_MX_LAYERS=0-7 P66_REHEARSAL_TOKENS=4 P66_REHEARSAL_WARM=2 \
  bash p66_run.sh > outer.log 2>&1
echo "full rc=$? seconds=$(( $(date +%s) - T0 )) exit=$(cat P66_EXIT_CODE.$NONCE 2>/dev/null) success=$(ls P66_SUCCESS.$NONCE 2>/dev/null) tp_done=$(ls TP_DONE.$NONCE 2>/dev/null)"
grep -E "^L |^M |MODE|TP_DONE|SKIPPED|incomplete" summary.txt outer.log | tail -30
rm -f $R/*.arena $R/*.dat 2>/dev/null
