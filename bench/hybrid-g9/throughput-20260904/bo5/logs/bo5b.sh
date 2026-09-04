#!/bin/bash
# bo5b: behind bo5's TP_DONE on the same box -- (1) #387 qkv fusion + fused rope-only fold on Mixtral and Granite, (2) the second-text
# attribution for Qwen3's `all` stack on c4val1 (int4 experts / calibrated attention / folds each against NF4). Cut: e4b integration-7
# @d090940ee10518872e4f8fd05dabe100f8dc1fa8 (main + #384 + #385 + #387) + gnf4 @587eb7aaf5618a045a92cf30d6a66bfb77507127.
set -uo pipefail
LANE=bo5; W=/root/$LANE; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook
say(){ echo "[$(date -u +%FT%TZ)] bo5b: $*"; }
say "waiting for TP_DONE"; while [ ! -f TP_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
say "install: e4b integration-7 @d090940ee10518872e4f8fd05dabe100f8dc1fa8 (+#387) + gnf4 @587eb7aaf5618a045a92cf30d6a66bfb77507127"
perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@d090940ee10518872e4f8fd05dabe100f8dc1fa8" > pip_bo5b.log 2>&1 || { say "PIP FAIL"; tail -3 pip_bo5b.log; touch TP2_DONE; exit 9; }
python -c "
from experts4bit_qlora.engines.qkv_fuse import _is_nonorm_attention, _fused_forward_nonorm
from experts4bit_qlora.engines.glue_r2 import _patch_attention_fused_rope_only, _patch_attention_rope_only
from experts4bit_qlora.engines.hot_residency import _swiglu_or, _combine_kernel
from experts4bit_qlora.engines.int4_experts import calibrate_expert_hessians
import experts4bit_qlora as e; print('bo5b tripwire OK: #387 + #385 + #384; e4b', e.__version__)" || { say "TRIPWIRE FAIL"; touch TP2_DONE; exit 9; }
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8(){ local TAG=$1 ARM=$2 EXP=$3 CA=$4 FU=$5 MID=$6 AR=$7; shift 7; read G R E <<<"$(fenv $FU)"
  say "K8 $TAG/$ARM (exp=$EXP calib=$CA fuse=$FU src=${PPLSRC:-wikitext} fq=${NOFQ:---no-fuse-qkv} $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 4200; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager ${NOFQ:---no-fuse-qkv} --ppl-source ${PPLSRC:-wikitext} --out $W/${TAG}_ppl_$ARM.json > run_${TAG}_ppl_$ARM.log 2>&1
  grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" run_${TAG}_ppl_$ARM.log | tail -3 | sed "s/^/    B5B /"; }
arm(){ local TAG=$1 ARM=$2 B=$3 EXP=$4 CA=$5 FU=$6 MID=$7 AR=$8; shift 8; read G R E <<<"$(fenv $FU)"
  say "arm $TAG/$ARM (B=$B exp=$EXP calib=$CA fuse=$FU fq=${NOFQ:---no-fuse-qkv} $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed ${NOFQ:---no-fuse-qkv} --out $W/${TAG}_b${B}_$ARM.json > run_${TAG}_b${B}_$ARM.log 2>&1
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|REFUSED|Error|glue r2|fused" run_${TAG}_b${B}_$ARM.log | tail -3 | sed "s/^/    B5B /"; }
bake(){ local MID=$1 TAG=$2; say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > bake2_$TAG.log 2>&1 || { echo "$TAG: BAKE FAILED"; tail -3 bake2_$TAG.log; return 1; }
  grep -aE "arena|BAKE" bake2_$TAG.log | tail -1; df -h /root | tail -1; }
# ---- Mixtral: still on disk from bo5. #387: fused qkv + the fused rope-only fold (drop --no-fuse-qkv) vs the unfused fold arm on this cut
MX=mistralai/Mixtral-8x7B-Instruct-v0.1; MA=$W/work_mixtral/nf4.arena
if [ -f $MA ]; then
  arm mixtral all_i7 1 1 1 all $MX $MA; NOFQ=" " arm mixtral all_fq 1 1 1 all $MX $MA; NOFQ=" " arm mixtral all_fq 16 1 1 all $MX $MA
  NOFQ=" " k8 mixtral all_fq 1 1 all $MX $MA
  echo "MIXTRAL FQ DONE"
else say "mixtral arena missing -- skipping the Mixtral #387 arms"; fi
rm -rf $W/work_mixtral /root/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1; say "freed mixtral ($(df -h /root | tail -1 | awk '{print $4}') free)"
# ---- Granite: re-bake (small). licensed NF4 stack: unfused+fold (control on this cut) vs fused qkv + fused fold (#387)
GR=ibm-granite/granite-3.1-3b-a800m-instruct; bake $GR granite && { GA=$W/work_granite/nf4.arena
  arm granite nf4_r12epi_i7 1 0 0 all $GR $GA; NOFQ=" " arm granite nf4_r12epi_fq2 1 0 0 all $GR $GA; NOFQ=" " arm granite nf4_r12epi_fq2 16 0 0 all $GR $GA
  NOFQ=" " k8 granite nf4_r12epi_fq2 0 0 all $GR $GA
  echo "GRANITE FQ DONE"; }
# ---- Qwen3: re-bake (60 GB download). The second-text attribution on c4val1, each component against NF4
Q=Qwen/Qwen3-30B-A3B; bake $Q qwen3 && { QA=$W/work_qwen3/nf4.arena
  PPLSRC=c4val1 k8 qwen3 int4exp_c4val 1 0 0 $Q $QA
  PPLSRC=c4val1 k8 qwen3 calib_c4val 1 1 0 $Q $QA
  PPLSRC=c4val1 k8 qwen3 folds_c4val 0 0 all $Q $QA
  PPLSRC=c4val1 k8 qwen3 int4folds_c4val 1 0 all $Q $QA
  echo "QWEN3 ATTRIBUTION DONE"; }
for f in run_*_ppl_*.log; do case $f in *_i7*|*_fq*|*_c4val.log) grep -a "K8_PPL" $f | tail -1 | sed "s/^/B5B SUMMARY $f /";; esac; done
for f in run_*_b1_*_i7.log run_*_b1_*_fq*.log run_*_b16_*_fq*.log; do [ -f $f ] && grep -aE "B1D_TIMED|BV3_" $f | tail -1 | sed "s/^/B5B SUMMARY $f /"; done
say "TP2_DONE"; touch TP2_DONE
