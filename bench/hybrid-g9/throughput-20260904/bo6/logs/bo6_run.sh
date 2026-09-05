#!/bin/bash
# Lane bo6 (2026-09-04): CLOSE THE GAP, not the gate. The registered K8 gate stays in ppl units. The int4-expert arms that
# failed their second text (Qwen3 `all` +0.063 ppl, Mixtral licensed stack +0.058 ppl on c4val1) are re-run with per-expert
# GPTQ calibration (#384 mechanism) in place of RTN. Cut: e4b integration-8 @db2a0703b803e50de1e3ac4d18b4705cb90c17fe
# (main 7dcc16f = 0.34.0 + #385 + #388, plus #384 and the GPU-solve knob) + gnf4 main @0b25d1389701bb793d60075e5b870212c848e33a.
set -uo pipefail
LANE=bo6; W=/root/$LANE; mkdir -p $W; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
say "install: gnf4 @0b25d1389701bb793d60075e5b870212c848e33a + e4b integration-8 @db2a0703b803e50de1e3ac4d18b4705cb90c17fe"
perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --prefer-binary --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@db2a0703b803e50de1e3ac4d18b4705cb90c17fe" > pip_e4b.log 2>&1; python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@0b25d1389701bb793d60075e5b870212c848e33a" "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@db2a0703b803e50de1e3ac4d18b4705cb90c17fe" "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors > pip.log 2>&1
rc=$?; echo "pip rc=$rc"; [ $rc -ne 0 ] && { tail -4 pip.log; echo "PIP FAIL"; touch TP_DONE; exit 9; }
python - <<'PYT' || { echo "TRIPWIRE FAIL"; touch TP_DONE; exit 9; }
from int4_b32 import rope_heads, reduce_partials, combine_rows, swiglu_rows
from gptq_pack import gptq_pack_int4_b32, HessianAccumulator
from experts4bit_qlora.engines.int4_experts import calibrate_expert_hessians, _ExpertHessianSink
from experts4bit_qlora.engines import int4_experts, hot_residency
import inspect, os
assert "E4B_INT4_GPTQ_DEVICE" in inspect.getsource(int4_experts), "gpu-solve knob missing"
assert hasattr(hot_residency, "_CALIB_SINK") and hasattr(hot_residency, "_swiglu_or"), "tap or swiglu path missing"
assert os.path.exists("/root/bo6/hook/usercustomize.py"), "hook missing"
import usercustomize  # noqa: F401
import experts4bit_qlora as e; print("bo6 tripwire OK: integration-8 (calibrated experts + gpu solve, swiglu/combine) + gnf4 main; e4b", e.__version__)
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX"; touch TP_DONE; exit 10; }
[ -s /root/.cache/huggingface/token ] || { echo "TOKEN MISSING"; touch TP_DONE; exit 8; }
df -h /root | tail -1
: > summary.txt
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
# k8 TAG ARM EXP CALIBATTN FUSE MODEL ARENA [extra env...]   (expert calibration is the extra env E4B_SERVE_EXP_INT4_CALIB=1)
k8(){ local TAG=$1 ARM=$2 EXP=$3 CA=$4 FU=$5 MID=$6 AR=$7; shift 7; read G R E <<<"$(fenv $FU)"
  say "K8 $TAG/$ARM (exp=$EXP calib=$CA fuse=$FU src=${PPLSRC:-wikitext} $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 5400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source ${PPLSRC:-wikitext} --out $W/${TAG}_ppl_$ARM.json > run_${TAG}_ppl_$ARM.log 2>&1
  grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" run_${TAG}_ppl_$ARM.log | tail -4 | sed "s/^/    /"; grep -aE "K8_PPL" run_${TAG}_ppl_$ARM.log | tail -1 | sed "s/^/SUMMARY run_${TAG}_ppl_$ARM.log /" >> summary.txt; }
arm(){ local TAG=$1 ARM=$2 B=$3 EXP=$4 CA=$5 FU=$6 MID=$7 AR=$8; shift 8; read G R E <<<"$(fenv $FU)"
  say "arm $TAG/$ARM (B=$B exp=$EXP calib=$CA fuse=$FU $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/${TAG}_b${B}_$ARM.json > run_${TAG}_b${B}_$ARM.log 2>&1
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|REFUSED|Error" run_${TAG}_b${B}_$ARM.log | tail -3 | sed "s/^/    /"; grep -aE "B1D_TIMED|BV3_" run_${TAG}_b${B}_$ARM.log | tail -1 | sed "s/^/SUMMARY run_${TAG}_b${B}_$ARM.log /" >> summary.txt; }
bake(){ local MID=$1 TAG=$2; [ -e "$W/work_$TAG/nf4.arena" ] && { say "bake $TAG: arena present, kept"; return 0; }; say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > bake_$TAG.log 2>&1 || { echo "$TAG: BAKE FAILED"; tail -3 bake_$TAG.log; return 1; }
  grep -aE "arena|BAKE" bake_$TAG.log | tail -1; df -h /root | tail -1; }
free_family(){ rm -rf $W/work_$1; rm -rf /root/.cache/huggingface/hub/models--${2}; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
# ---------------- Qwen3-30B-A3B: NF4 references on this cut, then calibrated int4 experts alone (the failing text) and the full stack on both texts
Q=Qwen/Qwen3-30B-A3B; bake $Q qwen3 && { QA=$W/work_qwen3/nf4.arena
  # nf4 / nf4_c4val: measured on attempt 2 (qwen3_ppl_nf4*.json), NF4 does not touch #384
  PPLSRC=c4val1 k8 qwen3 calibexp_c4val 1 0 0 $Q $QA E4B_SERVE_EXP_INT4_CALIB=1
  k8 qwen3 all_calibexp 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1
  PPLSRC=c4val1 k8 qwen3 all_calibexp_c4val 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1
  arm qwen3 all_calibexp 1 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1; arm qwen3 all_calibexp 16 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1
  echo "QWEN3 DONE"; }
free_family qwen3 Qwen--Qwen3-30B-A3B
# ---------------- Mixtral-8x7B: NF4 references on this cut, then the licensed stack with calibrated experts (no calibrated attention) on both texts
MX=mistralai/Mixtral-8x7B-Instruct-v0.1; bake $MX mixtral && { MA=$W/work_mixtral/nf4.arena
  k8 mixtral nf4 0 0 0 $MX $MA; PPLSRC=c4val1 k8 mixtral nf4_c4val 0 0 0 $MX $MA
  PPLSRC=c4val1 k8 mixtral lic_calibexp_c4val 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1
  k8 mixtral lic_calibexp 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1
  arm mixtral lic_calibexp 1 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1; arm mixtral lic_calibexp 16 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1
  echo "MIXTRAL DONE"; }
free_family mixtral mistralai--Mixtral-8x7B-Instruct-v0.1
say "TP_DONE"; touch TP_DONE
