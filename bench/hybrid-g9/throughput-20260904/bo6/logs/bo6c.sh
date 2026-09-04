#!/bin/bash
# bo6c: behind bo6b's TP2_DONE on the same box. The streamed (sequential) calibration flipped the calibrated-experts-alone verdict
# on Qwen3 c4val1 (+0.150 all-at-once -> -0.050 streamed). Re-score the FULL Qwen3 stack under the streamed method on both
# texts (it passed +0.035 under all-at-once), and repeat the streamed calibexp arm exactly, for calibration determinism.
set -uo pipefail
LANE=bo6; W=/root/$LANE; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda
say(){ echo "[$(date -u +%FT%TZ)] bo6c: $*"; }
say "waiting for TP2_DONE"; while [ ! -f TP2_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
say "reinstall e4b @d286dd52a8b23222479fb9de880bbdd00f6bf671 (damp knob)"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@ae9dc122e25216e8f8713c1631f5904e059efeb6" > pip_bo6c.log 2>&1 || { tail -3 pip_bo6c.log; say "PIP FAIL"; touch TP3_DONE; exit 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; touch TP3_DONE; exit 9; }
import inspect, os
from experts4bit_qlora.engines import int4_experts
assert "E4B_INT4_GPTQ_DAMP" in inspect.getsource(int4_experts) and "E4B_INT4_GPTQ_DEVICE" in inspect.getsource(int4_experts)
assert "E4B_CALIB_NSEQ" in open("/root/bo6/hook/usercustomize.py").read(), "hook v5 missing"
assert hasattr(int4_experts, "enable_serve_experts_int4_calibrated"), "streamed calibration missing"
assert "enable_serve_experts_int4_calibrated" in open("/root/bo6/hook/usercustomize.py").read(), "hook v6 missing"
print("bo6c tripwire OK: damp knob + hook v5")
PYT
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

Q=Qwen/Qwen3-30B-A3B
if bake $Q qwen3; then QA=$W/work_qwen3/nf4.arena
  export E4B_INT4_HESSIAN_BUDGET_GB=24
  # (1) calibration determinism: exact repeat of the streamed 16k-token arm
  PPLSRC=c4val1 k8 qwen3 calibexp_c4val_rep2 1 0 0 $Q $QA E4B_SERVE_EXP_INT4_CALIB=1
  # (2) the 64k-token size (best of the sweep, -0.211 ppl on c4val1) on the OUT-OF-DOMAIN text, alone
  k8 qwen3 calibexp_n128 1 0 0 $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128
  # (3) the full stack at 64k tokens on both texts -- the configuration that would ship
  k8 qwen3 all_calibexp_n128 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128
  PPLSRC=c4val1 k8 qwen3 all_calibexp_n128_c4val 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128
  echo "QWEN3 STREAMED STACK DONE"
else say "qwen3 bake failed"; fi
free_family qwen3 Qwen--Qwen3-30B-A3B
say "TP3_DONE"; touch TP3_DONE
