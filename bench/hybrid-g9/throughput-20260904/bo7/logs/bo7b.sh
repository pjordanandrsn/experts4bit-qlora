#!/bin/bash
# bo7b (P35 amendment 2, pre-registered 2026-09-05 06:05Z before TP_DONE): the LICENSED Qwen3 stack's speed on the bo7 box, same session.
# bo6c licensed the streamed 64k-token pack (E4B_CALIB_NSEQ=128); bo7's calibrated arms ran the hook's 16k default. Two arms, B=1 and B=16,
# read against bo7's own qwen3 nf4 arms (this box, this session). Runs after TP_DONE; touches TP2_DONE. Same cut, hook v6, install untouched.
set -uo pipefail
LANE=bo7; W=/root/$LANE; mkdir -p $W; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
# k8 TAG ARM EXP CALIBATTN FUSE MODEL ARENA [extra env...]   (expert calibration is the extra env E4B_SERVE_EXP_INT4_CALIB=1)
arm(){ local TAG=$1 ARM=$2 B=$3 EXP=$4 CA=$5 FU=$6 MID=$7 AR=$8; shift 8; read G R E <<<"$(fenv $FU)"
  local AL=3600; case "$*" in *E4B_SERVE_EXP_INT4_CALIB=1*) AL=5400;; esac
  say "arm $TAG/$ARM (B=$B exp=$EXP calib=$CA fuse=$FU $* alarm=$AL)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/${TAG}_b${B}_$ARM.json > run_${TAG}_b${B}_$ARM.log 2>&1
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|REFUSED|Error" run_${TAG}_b${B}_$ARM.log | tail -3 | sed "s/^/    /"; grep -aE "B1D_TIMED|BV3_" run_${TAG}_b${B}_$ARM.log | tail -1 | sed "s/^/SUMMARY run_${TAG}_b${B}_$ARM.log /" >> summary.txt; }
bake(){ local MID=$1 TAG=$2; [ -e "$W/work_$TAG/nf4.arena" ] && { say "bake $TAG: arena present, kept"; return 0; }; say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > bake_$TAG.log 2>&1 || { echo "$TAG: BAKE FAILED"; tail -3 bake_$TAG.log; return 1; }
  grep -aE "arena|BAKE" bake_$TAG.log | tail -1; df -h /root | tail -1; }
free_family(){ rm -rf $W/work_$1; rm -rf /root/.cache/huggingface/hub/models--${2}; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }

CAL="E4B_SERVE_EXP_INT4_CALIB=1"
say "bo7b: licensed Qwen3 stack (streamed 64k pack) speed arms"
Q=Qwen/Qwen3-30B-A3B; bake $Q qwen3 && { QA=$W/work_qwen3/nf4.arena
  for B in 1 16; do arm qwen3 calibexp_all_n128 $B 1 1 all $Q $QA $CAL E4B_CALIB_NSEQ=128; done; echo "QWEN3 N128 DONE"; }
free_family qwen3 Qwen--Qwen3-30B-A3B
say "BO7B COMPLETE"; touch TP2_DONE
