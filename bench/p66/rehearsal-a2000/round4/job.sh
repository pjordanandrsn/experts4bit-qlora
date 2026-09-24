#!/bin/bash
# P66 rehearsal, round 4: bench/p66/p66_run.sh ITSELF, end to end on the A2000, with its documented rehearsal
# overrides (local snapshots, the prebaked OLMoE arena, no install, A2000 minimums). NOT a reading. Round 4 = the FINAL staged tree.
set -u
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/w/hfcache
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$FREE" -ge 8500 ] || { echo "ABORT: only ${FREE} MiB free on the shared A2000"; exit 75; }
R=/root/p66; rm -rf $R/* 2>/dev/null; mkdir -p $R
cp /w/src/e4b/bench/p66/p66_run.sh /w/src/e4b/bench/p66/p66_census.py /w/src/e4b/bench/p66/p66_reduce.py \
   /w/src/e4b/bench/p66/p66_step.py /w/src/e4b/bench/p66/staged.sha256 /w/src/e4b/bench/p39/k8_bake.py \
   /w/src/e4b/bench/hybrid-g9/step_decomp.py /w/src/e4b/bench/hybrid-g9/f1/step_budget.py $R/
cd $R
NONCE=$(python -c 'import secrets; print(secrets.token_hex(32))')
env P66_RUN_ID=p66-rehearsal-a2000 P66_RUN_NONCE=$NONCE P66_DEADLINE_EPOCH=$(( $(date +%s) + 5400 )) P66_INSTANCE_ID=nas-a2000 \
  E4B_SHA=94842a224f0691a008d25a509bd220e4dcb49ca7 GNF4_SHA=66d41c821bb29e37551a721f91b16b8ecbb49ed9 \
  P66_REHEARSAL_CLASS=A2000 P66_REHEARSAL_MIN_VRAM_MB=8500 P66_REHEARSAL_MIN_DISK_GB=50 P66_REHEARSAL_MIN_RAM_GB=16 \
  P66_REHEARSAL_PIN_GIB=4 P66_REHEARSAL_SKIP_INSTALL=1 P66_REHEARSAL_GNF4_SRC=/w/src/gnf4 \
  P66_REHEARSAL_QWEN_SNAPSHOT=/models/OLMoE-1B-7B-0924 P66_REHEARSAL_NF4_FAMILY=olmoe-1b-7b \
  P66_REHEARSAL_NF4_ARENA=/w/work/olmoe/nf4.arena P66_REHEARSAL_GPTOSS_SNAPSHOT=/models/gpt-oss-20b \
  P66_REHEARSAL_MX_LAYERS=0-7 P66_REHEARSAL_TOKENS=4 P66_REHEARSAL_WARM=2 \
  bash p66_run.sh > outer.log 2>&1
echo "runner rc=$? exit-code-file=$(cat P66_EXIT_CODE.$NONCE 2>/dev/null) success=$(ls P66_SUCCESS.$NONCE 2>/dev/null) tp_done=$(ls TP_DONE.$NONCE 2>/dev/null)"
grep -v "^\s*$" outer.log | tail -80
rm -f $R/*.arena $R/gnf4-nvme-bench* 2>/dev/null; ls -la $R
