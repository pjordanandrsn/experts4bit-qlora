#!/bin/bash
# bo3e: behind bo3d -- the calibrated OUTPUT HEAD arms on Qwen3 (#373), which bo3b lost to the install no-op.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3e: $*"; }
say "waiting for BO3D_DONE"; while [ ! -f BO3D_DONE ]; do sleep 30; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v2
python -c "from experts4bit_qlora.engines.int4_attn_calib import _LM_HEAD_ENV; print('bo3e tripwire OK: head lane present')" || { say "TRIPWIRE FAIL: head lane missing"; touch BO3E_DONE; exit 9; }
MID=Qwen/Qwen3-30B-A3B; TAG=qwen3; AR=$W/work_$TAG/nf4.arena
[ -f $AR ] || { say "no qwen3 arena"; touch BO3E_DONE; exit 2; }
fenv(){ case "$1" in 0) echo "0 0 0";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8h(){ read G R E <<<"$(fenv $5)"; say "K8 $TAG/$1 (exp=$2 calib=$3 head=$4 fuse=$5)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$2 E4B_SERVE_ATTN_INT4_CALIB=$3 E4B_SERVE_LMHEAD_INT4_CALIB=$4 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/${TAG}_ppl_$1.json > run_${TAG}_ppl_$1.log 2>&1
  grep -aE "K8_PPL|ATTNINT4|REFUSED|Error" run_${TAG}_ppl_$1.log | tail -2 | sed "s/^/    HEAD /"; }
armh(){ read G R E <<<"$(fenv $6)"; say "arm $TAG/$1 (B=$2 exp=$3 calib=$4 head=$5 fuse=$6)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_SERVE_LMHEAD_INT4_CALIB=$5 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $2 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/${TAG}_$1.json > run_${TAG}_$1.log 2>&1
  grep -aE "B1D_TIMED|BV3_|ATTNINT4|REFUSED|Error" run_${TAG}_$1.log | tail -2 | sed "s/^/    HEAD /"; }
k8h head 0 0 1 0; k8h allhead 1 1 1 all
armh b1_head 1 0 0 1 0; armh b1_allhead 1 1 1 1 all
python - <<'PY'
import json, glob, os
print("HEAD SUMMARY (qwen3)")
for f in sorted(glob.glob("/root/bo3/qwen3_*head*.json")):
    try:
        d=json.load(open(f)); s=d.get("step_ms_clean"); n=d.get("mean_nll")
        print("  %-26s %s" % (os.path.basename(f), ("%.1f tok/s" % (1000/s)) if s else ("nll %.5f" % n if n else "?")))
    except Exception as e: print("  ", f, "unreadable", e)
PY
say "BO3E_DONE"; touch BO3E_DONE
