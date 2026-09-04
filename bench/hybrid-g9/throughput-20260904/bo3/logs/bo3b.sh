#!/bin/bash
# bo3b: behind the bo3 lane -- the calibrated int4 OUTPUT HEAD (e4b#373) on Qwen3 and Gemma-4, whose arenas bo3 baked.
# Head-only and head-beside-the-full-stack, K8 (2048) and B=1; the uncalibrated head's +0.18 ppl is the number to beat.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3b: $*"; }
say "waiting for TP_DONE"; while [ ! -f TP_DONE ]; do sleep 30; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v2
say "install: e4b @claude/buildout-integration-3 (= integration-2 + #373)"
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@claude/buildout-integration-3" > pip_bo3b.log 2>&1 || { say "pip FAILED"; tail -3 pip_bo3b.log; touch BO3B_DONE; exit 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; touch BO3B_DONE; exit 9; }
from experts4bit_qlora.engines.int4_attn_calib import _LM_HEAD_ENV, _int4_targets, enable_from_env
import os; assert os.path.exists("/root/bo3/hook_v2/usercustomize.py"); import usercustomize  # noqa
import experts4bit_qlora as e; print("bo3b tripwire OK: head lane present; e4b", e.__version__)
PYT
fenv(){ case "$1" in 0) echo "0 0 0";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8h(){ # $1 tag $2 arm $3 exp $4 calib $5 head $6 fuse $7 model $8 arena
  read G R E <<<"$(fenv $6)"; say "K8 $1/$2 (exp=$3 calib=$4 head=$5 fuse=$6 steps=2048)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$7" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_SERVE_LMHEAD_INT4_CALIB=$5 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$7" --arena "$8" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/$1_ppl_$2.json > run_$1_ppl_$2.log 2>&1
  grep -aE "K8_PPL|ATTNINT4|REFUSED|Error" run_$1_ppl_$2.log | tail -2 | sed "s/^/    HEAD /"
}
armh(){ # $1 tag $2 arm $3 batch $4 exp $5 calib $6 head $7 fuse $8 model $9 arena
  read G R E <<<"$(fenv $7)"; say "arm $1/$2 (B=$3 exp=$4 calib=$5 head=$6 fuse=$7)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$8" E4B_SERVE_EXP_INT4=$4 E4B_SERVE_ATTN_INT4_CALIB=$5 E4B_SERVE_LMHEAD_INT4_CALIB=$6 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$8" --arena "$9" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $3 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/$1_$2.json > run_$1_$2.log 2>&1
  grep -aE "B1D_TIMED|BV3_|ATTNINT4|REFUSED|Error" run_$1_$2.log | tail -2 | sed "s/^/    HEAD /"
}
for spec in "Qwen/Qwen3-30B-A3B qwen3" "google/gemma-4-26B-A4B-it gemma4"; do
  set -- $spec; MID=$1; TAG=$2; AR=$W/work_$TAG/nf4.arena
  [ -f $AR ] || { say "$TAG: no arena from bo3 -- skipped"; continue; }
  say "=============== $TAG head arms ==============="
  k8h  $TAG head     0 0 1 0   "$MID" $AR
  k8h  $TAG allhead  1 1 1 all "$MID" $AR
  armh $TAG b1_head    1 0 0 1 0   "$MID" $AR
  armh $TAG b1_allhead 1 1 1 1 all "$MID" $AR
done
python - <<'PY'
import json, glob, os
print("HEAD SUMMARY")
for f in sorted(glob.glob("/root/bo3/*_ppl_*head*.json") + glob.glob("/root/bo3/*_b1_*head*.json")):
    try:
        d=json.load(open(f)); s=d.get("step_ms_clean"); n=d.get("mean_nll")
        print("  %-28s %s" % (os.path.basename(f), ("%.1f tok/s" % (1000/s)) if s else ("nll %.5f" % n if n else "?")))
    except Exception as e: print("  ", f, "unreadable", e)
PY
say "BO3B_DONE"; touch BO3B_DONE
