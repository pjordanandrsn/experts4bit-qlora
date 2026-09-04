#!/bin/bash
# bo6b (v2): behind bo6's TP_DONE on the same box. (1) INSTRUMENT control: the same NF4 arm repeated on the same box and cut
# (bo5 and bo6 disagree by 0.006 nats on the identical c4val1 window). (2) CLOSE-THE-GAP follow-up: Qwen3 calibrated experts
# read +0.150 ppl on c4val1 with the hook's fixed 16k-token calibration set (~1k rows per expert against a 2048-wide Hessian);
# sweep the set size (n_seq 128, 512) and the damping (0.1 at n_seq 32); then one Mixtral arm at n_seq 128.
# The e4b cut is reinstalled at (streamed calibration: Mixtral's Hessians no longer pile up on the host) d286dd52a8b23222479fb9de880bbdd00f6bf671 first: it adds only the E4B_INT4_GPTQ_DAMP env (default = the kernel's 0.01), numerically
# identical to attempt 3's db2a070 for every arm that does not set it.
set -uo pipefail
LANE=bo6; W=/root/$LANE; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda
say(){ echo "[$(date -u +%FT%TZ)] bo6b: $*"; }
say "waiting for TP_DONE"; while [ ! -f TP_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
say "reinstall e4b @d286dd52a8b23222479fb9de880bbdd00f6bf671 (damp knob)"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@ae9dc122e25216e8f8713c1631f5904e059efeb6" > pip_bo6b.log 2>&1 || { tail -3 pip_bo6b.log; say "PIP FAIL"; touch TP2_DONE; exit 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; touch TP2_DONE; exit 9; }
import inspect, os
from experts4bit_qlora.engines import int4_experts
assert "E4B_INT4_GPTQ_DAMP" in inspect.getsource(int4_experts) and "E4B_INT4_GPTQ_DEVICE" in inspect.getsource(int4_experts)
assert "E4B_CALIB_NSEQ" in open("/root/bo6/hook/usercustomize.py").read(), "hook v5 missing"
assert hasattr(int4_experts, "enable_serve_experts_int4_calibrated"), "streamed calibration missing"
assert "enable_serve_experts_int4_calibrated" in open("/root/bo6/hook/usercustomize.py").read(), "hook v6 missing"
print("bo6b tripwire OK: damp knob + hook v5")
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
  # (1) instrument control: NF4 repeated, both texts
  PPLSRC=c4val1 k8 qwen3 nf4_c4val_rep1 0 0 0 $Q $QA; PPLSRC=c4val1 k8 qwen3 nf4_c4val_rep2 0 0 0 $Q $QA
  k8 qwen3 nf4_rep1 0 0 0 $Q $QA
  # (2) the calibrated arm repeated as-is (n_seq 32, damp 0.01), then the sweep
  PPLSRC=c4val1 k8 qwen3 calibexp_c4val_rep1 1 0 0 $Q $QA E4B_SERVE_EXP_INT4_CALIB=1
  PPLSRC=c4val1 k8 qwen3 calibexp_d01_c4val 1 0 0 $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_INT4_GPTQ_DAMP=0.1
  PPLSRC=c4val1 k8 qwen3 calibexp_n128_c4val 1 0 0 $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128
  PPLSRC=c4val1 k8 qwen3 calibexp_n512_c4val 1 0 0 $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=512
  echo "QWEN3 SWEEP DONE"
else say "qwen3 bake failed -- no repeat controls or sweep"; fi
free_family qwen3 Qwen--Qwen3-30B-A3B
MX=mistralai/Mixtral-8x7B-Instruct-v0.1
if bake $MX mixtral; then MA=$W/work_mixtral/nf4.arena
  # bo6's Mixtral calibrated arms were OOM-killed by the 170 GiB container (all-layer Hessians); re-run them streamed, 8 GiB per pass
  export E4B_INT4_HESSIAN_BUDGET_GB=8
  PPLSRC=c4val1 k8 mixtral lic_calibexp_c4val 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1
  k8 mixtral lic_calibexp 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1
  arm mixtral lic_calibexp 1 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1; arm mixtral lic_calibexp 16 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1
  PPLSRC=c4val1 k8 mixtral lic_calibexp_n128_c4val 1 0 all $MX $MA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128
  echo "MIXTRAL SWEEP DONE"
else say "mixtral bake failed -- no Mixtral sweep arm"; fi
free_family mixtral mistralai--Mixtral-8x7B-Instruct-v0.1
say "TP2_DONE"; touch TP2_DONE
