#!/bin/bash
# bo5c: behind bo5b's TP2_DONE on the same box -- the SECOND TEXT for Mixtral's LICENSED stack (int4 experts + r1 + r2 rope-only fold +
# router epilogue, no calibrated attention, no fused qkv) on the integration-7 cut bo5b installed, plus its B=1/B=16 arms and an NF4
# wikitext control. Mixtral was freed by bo5b, so this re-downloads and re-bakes it first.
set -uo pipefail
LANE=bo5; W=/root/$LANE; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook
say(){ echo "[$(date -u +%FT%TZ)] bo5c: $*"; }
say "waiting for TP2_DONE"; while [ ! -f TP2_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
python -c "
from experts4bit_qlora.engines.glue_r2 import _patch_attention_rope_only
from experts4bit_qlora.engines.hot_residency import _swiglu_or, _combine_kernel
import experts4bit_qlora as e; print('bo5c tripwire OK: integration-7 still installed; e4b', e.__version__)" || { say "TRIPWIRE FAIL"; touch TP3_DONE; exit 9; }
# make room: bo5b's re-bakes are done by now
rm -rf $W/work_qwen3 /root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B 2>/dev/null
say "disk after free: $(df -h /root | tail -1 | awk '{print $4}') free"
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8(){ local TAG=$1 ARM=$2 EXP=$3 CA=$4 FU=$5 MID=$6 AR=$7; shift 7; read G R E <<<"$(fenv $FU)"
  say "K8 $TAG/$ARM (exp=$EXP calib=$CA fuse=$FU src=${PPLSRC:-wikitext} fq=${NOFQ:---no-fuse-qkv} $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 4200; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager ${NOFQ:---no-fuse-qkv} --ppl-source ${PPLSRC:-wikitext} --out $W/${TAG}_ppl_$ARM.json > run_${TAG}_ppl_$ARM.log 2>&1
  grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" run_${TAG}_ppl_$ARM.log | tail -3 | sed "s/^/    B5C /"; }
arm(){ local TAG=$1 ARM=$2 B=$3 EXP=$4 CA=$5 FU=$6 MID=$7 AR=$8; shift 8; read G R E <<<"$(fenv $FU)"
  say "arm $TAG/$ARM (B=$B exp=$EXP calib=$CA fuse=$FU fq=${NOFQ:---no-fuse-qkv} $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed ${NOFQ:---no-fuse-qkv} ${CENSUS:+--replay-profile-out $W/census_${TAG}_$ARM.txt} --out $W/${TAG}_b${B}_$ARM.json > run_${TAG}_b${B}_$ARM.log 2>&1
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|REFUSED|Error|glue r2|fused" run_${TAG}_b${B}_$ARM.log | tail -3 | sed "s/^/    B5C /"; }
bake(){ local MID=$1 TAG=$2; say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > bake3_$TAG.log 2>&1 || { echo "$TAG: BAKE FAILED"; tail -3 bake3_$TAG.log; return 1; }
  grep -aE "arena|BAKE" bake3_$TAG.log | tail -1; df -h /root | tail -1; }
# Granite is still on disk from bo5b: the missing B=16 unfused control, a second B=1 pair, and a kernel census of unfused vs fused (#387)
GR=ibm-granite/granite-3.1-3b-a800m-instruct; GA=$W/work_granite/nf4.arena
if [ -e "$GA" ]; then
  arm granite nf4_r12epi_i7 16 0 0 all $GR $GA
  arm granite nf4_r12epi_i7b 1 0 0 all $GR $GA; NOFQ=" " arm granite nf4_r12epi_fq2b 1 0 0 all $GR $GA
  CENSUS=1 arm granite nf4_r12epi_i7_cen 1 0 0 all $GR $GA; CENSUS=1 NOFQ=" " arm granite nf4_r12epi_fq2_cen 1 0 0 all $GR $GA
  echo "GRANITE CENSUS DONE"
else say "granite arena missing -- skipping the Granite census"; fi
rm -rf $W/work_granite /root/.cache/huggingface/hub/models--ibm-granite--granite-3.1-3b-a800m-instruct; say "freed granite"
MX=mistralai/Mixtral-8x7B-Instruct-v0.1
if bake $MX mixtral; then MA=$W/work_mixtral/nf4.arena
  k8 mixtral nf4_i7 0 0 0 $MX $MA                       # NF4 wikitext control on this cut (P30: 1.18048)
  k8 mixtral lic 1 0 all $MX $MA                        # licensed stack, wikitext
  PPLSRC=c4val1 k8 mixtral lic_c4val 1 0 all $MX $MA    # licensed stack, SECOND text (NF4 c4val1 = 8.25079 from bo5)
  arm mixtral lic 1 1 0 all $MX $MA; arm mixtral lic 16 1 0 all $MX $MA
  # #387 on the LICENSED stack (calib=0, so the q/k/v children are plain Linear): fused qkv + fused rope-only fold
  NOFQ=" " arm mixtral lic_fq 1 1 0 all $MX $MA; NOFQ=" " arm mixtral lic_fq 16 1 0 all $MX $MA
  NOFQ=" " k8 mixtral lic_fq 1 0 all $MX $MA
  echo "MIXTRAL-LIC DONE"
else say "mixtral bake failed -- no licensed-stack second text this lane"; fi
say "TP3_DONE"; touch TP3_DONE
