#!/bin/bash
# Lane bo5 (2026-09-04): the optimisation pass, second lane. Cut: e4b integration-6 @0535930324e5d11c35966ae0569f3c519a684355 (main 0.34.0 + #372 + #384 calibrated experts + #385 swiglu/combine)
# + gnf4 main @587eb7aaf5618a045a92cf30d6a66bfb77507127 (0.29.0 + combine_rows). Arms: Qwen3 all (+A/B of the two glue fusions, census), Granite licensed stack + CALIBRATED int4
# experts (the gate) + fused-qkv variant, gpt-oss MXFP4 store + GEMV with NF4 kept (B=1/B=16), Mixtral all (rotary-only fold engages) vs no-R2.
set -uo pipefail
LANE=bo5; W=/root/$LANE; mkdir -p $W; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
say "install: gnf4 @587eb7aaf5618a045a92cf30d6a66bfb77507127 + e4b @0535930324e5d11c35966ae0569f3c519a684355"
perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@587eb7aaf5618a045a92cf30d6a66bfb77507127" "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@0535930324e5d11c35966ae0569f3c519a684355" "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken > pip.log 2>&1
rc=$?; echo "pip rc=$rc"; [ $rc -ne 0 ] && { tail -4 pip.log; echo "PIP FAIL"; touch TP_DONE; exit 9; }
python - <<'PYT' || { echo "TRIPWIRE FAIL"; touch TP_DONE; exit 9; }
from int4_b32 import rope_heads, reduce_partials, combine_rows, swiglu_rows
from mxfp4_grouped import gemv_mxfp4_b32
from experts4bit_qlora.engines.int4_experts import calibrate_expert_hessians, _ExpertHessianSink
from experts4bit_qlora.engines.hot_residency import _swiglu_or, _combine_kernel, _MXFP4_GEMV_ROWS
from experts4bit_qlora.engines.glue_r2 import _patch_attention_rope_only, _patch_attention_unfused
import os; assert os.path.exists("/root/bo5/hook/usercustomize.py"), "hook missing"
import usercustomize  # noqa: F401
import experts4bit_qlora as e; print("bo5 tripwire OK: integration-6 (calib experts, swiglu/combine, mxfp4 route) + gnf4 0.29.0+combine; e4b", e.__version__)
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX"; touch TP_DONE; exit 10; }
[ -s /root/.cache/huggingface/token ] || { echo "TOKEN MISSING"; touch TP_DONE; exit 8; }
df -h /root | tail -1
: > summary.txt
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
# k8 TAG ARM EXP CALIBATTN FUSE MODEL ARENA [extra env...]
k8(){ local TAG=$1 ARM=$2 EXP=$3 CA=$4 FU=$5 MID=$6 AR=$7; shift 7; read G R E <<<"$(fenv $FU)"
  say "K8 $TAG/$ARM (exp=$EXP calib=$CA fuse=$FU $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 4200; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager ${NOFQ:---no-fuse-qkv} --ppl-source ${PPLSRC:-wikitext} --out $W/${TAG}_ppl_$ARM.json > run_${TAG}_ppl_$ARM.log 2>&1
  grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" run_${TAG}_ppl_$ARM.log | tail -3 | sed "s/^/    /"; }
# arm TAG ARM BATCH EXP CALIBATTN FUSE MODEL ARENA [extra env...]
arm(){ local TAG=$1 ARM=$2 B=$3 EXP=$4 CA=$5 FU=$6 MID=$7 AR=$8; shift 8; read G R E <<<"$(fenv $FU)"
  say "arm $TAG/$ARM (B=$B exp=$EXP calib=$CA fuse=$FU $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed ${NOFQ:---no-fuse-qkv} ${CENSUS:+--replay-profile-out $W/census_${TAG}_$ARM.txt} --out $W/${TAG}_b${B}_$ARM.json > run_${TAG}_b${B}_$ARM.log 2>&1
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|REPLAY_PROFILE|REFUSED|Error" run_${TAG}_b${B}_$ARM.log | tail -3 | sed "s/^/    /"; }
bake(){ local MID=$1 TAG=$2; say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > bake_$TAG.log 2>&1 || { echo "$TAG: BAKE FAILED"; tail -3 bake_$TAG.log; return 1; }
  grep -aE "arena|BAKE" bake_$TAG.log | tail -1; df -h /root | tail -1; }
free_family(){ rm -rf $W/work_$1; rm -rf /root/.cache/huggingface/hub/models--${2}; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
# ---------------- Qwen3: the reference, with the glue fusions on/off and a census
Q=Qwen/Qwen3-30B-A3B; bake $Q qwen3 && { QA=$W/work_qwen3/nf4.arena
  k8 qwen3 all 1 1 all $Q $QA; arm qwen3 all 1 1 1 all $Q $QA; arm qwen3 all 16 1 1 all $Q $QA
  PPLSRC=c4val1 k8 qwen3 all_c4val 1 1 all $Q $QA; PPLSRC=c4val1 k8 qwen3 nf4_c4val 0 0 0 $Q $QA
  arm qwen3 all_noglue 1 1 1 all $Q $QA E4B_FUSE_SWIGLU=0 E4B_FUSE_COMBINE=0
  CENSUS=1 arm qwen3 all_cen 1 1 1 all $Q $QA; CENSUS=1 arm qwen3 all_noglue_cen 1 1 1 all $Q $QA E4B_FUSE_SWIGLU=0 E4B_FUSE_COMBINE=0
  echo "QWEN3 DONE"; }
free_family qwen3 Qwen--Qwen3-30B-A3B
# ---------------- Granite: licensed stack, CALIBRATED int4 experts (the gate), fused-qkv variant
GR=ibm-granite/granite-3.1-3b-a800m-instruct; bake $GR granite && { GA=$W/work_granite/nf4.arena
  k8 granite nf4_r12epi 0 0 all $GR $GA; arm granite nf4_r12epi 1 0 0 all $GR $GA; arm granite nf4_r12epi 16 0 0 all $GR $GA
  k8 granite calibexp_r12epi 1 0 all $GR $GA E4B_SERVE_EXP_INT4_CALIB=1; arm granite calibexp_r12epi 1 1 0 all $GR $GA E4B_SERVE_EXP_INT4_CALIB=1; arm granite calibexp_r12epi 16 1 0 all $GR $GA E4B_SERVE_EXP_INT4_CALIB=1
  PPLSRC=c4val1 k8 granite calibexp_r12epi_c4val 1 0 all $GR $GA E4B_SERVE_EXP_INT4_CALIB=1
  PPLSRC=c4val1 k8 granite nf4_r12epi_c4val 0 0 all $GR $GA
  k8 granite rtnexp_r12epi 1 0 all $GR $GA
  NOFQ=" " arm granite nf4_r12epi_fq 1 0 0 all $GR $GA; NOFQ=" " k8 granite nf4_r12epi_fq 0 0 all $GR $GA
  echo "GRANITE DONE"; }
# ---------------- gpt-oss: MXFP4 store + decode GEMV with the NF4 stacks kept
GO=openai/gpt-oss-20b; bake $GO gptoss && { OA=$W/work_gptoss/nf4.arena
  arm gptoss r12 1 0 0 r12 $GO $OA; arm gptoss r12 16 0 0 r12 $GO $OA
  arm gptoss store_r12 1 1 0 r12 $GO $OA E4B_INT4_KEEP_NF4=1; arm gptoss store_r12 16 1 0 r12 $GO $OA E4B_INT4_KEEP_NF4=1
  CENSUS=1 arm gptoss store_r12_cen 1 1 0 r12 $GO $OA E4B_INT4_KEEP_NF4=1
  echo "GPTOSS DONE"; }
free_family granite ibm-granite--granite-3.1-3b-a800m-instruct; free_family gptoss openai--gpt-oss-20b
# ---------------- Mixtral: the rotary-only fold on the second norm-less family
MX=mistralai/Mixtral-8x7B-Instruct-v0.1; bake $MX mixtral && { MA=$W/work_mixtral/nf4.arena
  k8 mixtral all 1 1 all $MX $MA; arm mixtral all 1 1 1 all $MX $MA; arm mixtral all 16 1 1 all $MX $MA
  PPLSRC=c4val1 k8 mixtral all_c4val 1 1 all $MX $MA; PPLSRC=c4val1 k8 mixtral nf4_c4val 0 0 0 $MX $MA
  arm mixtral all_nor2 1 1 1 r1epi $MX $MA; arm mixtral all_nor2 16 1 1 r1epi $MX $MA
  echo "MIXTRAL DONE"; }
for f in run_*_ppl_*.log; do grep -a "K8_PPL" $f | tail -1 | sed "s/^/SUMMARY $f /"; done
for f in run_*_b1_*.log run_*_b16_*.log; do grep -aE "B1D_TIMED|BV3_" $f | tail -1 | sed "s/^/SUMMARY $f /"; done
say "TP_DONE"; touch TP_DONE
