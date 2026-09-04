#!/bin/bash
# bo3c: behind bo3b -- gpt-oss with the NATIVE MXFP4 expert store (e4b#372, integration-2 @87f038e): K8 int4exp/stack,
# B=1 int4exp/stack, B=16 int4exp/stack. The gate: K8 vs NF4 6.33544 against the 0.0176-nat floor.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3c: $*"; }
say "waiting for BO3B_DONE"; while [ ! -f BO3B_DONE ]; do sleep 30; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v2
say "install: e4b @87f038e (integration-2 + native MXFP4 store)"
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@87f038e" > pip_bo3c.log 2>&1 || { say "pip FAILED"; tail -3 pip_bo3c.log; touch BO3C_DONE; exit 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; touch BO3C_DONE; exit 9; }
from experts4bit_qlora.engines.int4_experts import _mxfp4_store_layout
from mxfp4_grouped import gemm_mxfp4_grouped
import experts4bit_qlora as e; print("bo3c tripwire OK: native MXFP4 store + kernel; e4b", e.__version__)
PYT
MID=openai/gpt-oss-20b; TAG=gptoss; AR=$W/work_$TAG/nf4.arena
[ -f $AR ] || { say "no gptoss arena"; touch BO3C_DONE; exit 2; }
fenv(){ case "$1" in 0) echo "0 0 0";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8m(){ # $1 arm $2 exp $3 fuse
  read G R E <<<"$(fenv $3)"; say "K8 $TAG/mx_$1 (exp=$2 fuse=$3 steps=2048)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$2 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/${TAG}_ppl_mx_$1.json > run_${TAG}_ppl_mx_$1.log 2>&1
  grep -aE "K8_PPL|INT4EXP|MXFP4|REFUSED|Error" run_${TAG}_ppl_mx_$1.log | tail -2 | sed "s/^/    MX /"
}
armm(){ # $1 arm $2 batch $3 exp $4 fuse
  read G R E <<<"$(fenv $4)"; say "arm $TAG/mx_$1 (B=$2 exp=$3 fuse=$4)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $2 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/${TAG}_mx_$1.json > run_${TAG}_mx_$1.log 2>&1
  grep -aE "B1D_TIMED|BV3_|INT4EXP|REFUSED|Error" run_${TAG}_mx_$1.log | tail -2 | sed "s/^/    MX /"
}
k8m  int4exp 1 0;   k8m  stack 1 r1epi
armm b1_int4exp 1 1 0;  armm b1_stack 1 1 r1epi
armm b16_int4exp 16 1 0; armm b16_stack 16 1 r1epi
python - <<'PY'
import json, glob, os
print("MX SUMMARY (gpt-oss native MXFP4 store)")
for f in sorted(glob.glob("/root/bo3/gptoss_*mx_*.json")):
    try:
        d=json.load(open(f)); s=d.get("step_ms_clean"); a=d.get("aggregate_tok_s"); n=d.get("mean_nll")
        print("  %-30s %s" % (os.path.basename(f), ("%.1f tok/s" % (1000/s)) if s else (("B16 %.1f" % a) if a else ("nll %.5f" % n if n else "?"))))
    except Exception as e: print("  ", f, "unreadable", e)
PY
say "BO3C_DONE"; touch BO3C_DONE
