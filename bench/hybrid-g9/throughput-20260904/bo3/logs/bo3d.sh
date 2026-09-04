#!/bin/bash
# bo3d: Gemma-4 redo after the disk filled (Mixtral + Granite freed): download, bake, the per-fusion arms of the lane
# (integration-2 @87f038e already installed), then the calibrated output-head arms (integration-3). Qwen3 is dropped
# from this box (its regression ran on bo; head arms on Qwen3 go to a later lane).
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3d: $*"; }
say "waiting for BO3C_DONE"; while [ ! -f BO3C_DONE ]; do sleep 30; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook
MID=google/gemma-4-26B-A4B-it; TAG=gemma4
say "free space: $(df -h /root | tail -1 | awk '{print $4}')"
python - "$MID" <<'PYD' > dl_${TAG}_redo.log 2>&1
import os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
for a in range(6):
    try:
        print("OK", snapshot_download(sys.argv[1], allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken"], ignore_patterns=["original/*","consolidated*"], max_workers=4)); break
    except Exception as e:
        print("retry", a, repr(e)[:120], flush=True); time.sleep(30)
PYD
grep -q "^OK " dl_${TAG}_redo.log || { say "DOWNLOAD FAILED: $(tail -1 dl_${TAG}_redo.log | cut -c1-120)"; touch BO3D_DONE; exit 2; }
say "bake $TAG"
K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > bake_$TAG.log 2>&1
python -c "import json,sys;d=json.load(open('$W/work_$TAG/bake.json'));sys.exit(0 if d.get('status')=='OK' else 1)" 2>/dev/null || { say "BAKE FAILED"; grep -aiE 'error|cuda' bake_$TAG.log | tail -2; touch BO3D_DONE; exit 3; }
AR=$W/work_$TAG/nf4.arena
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8(){ read G R E <<<"$(fenv $5)"; say "K8 $1/$2 (exp=$3 calib=$4 fuse=$5 head=${HEADF:-0})"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$6" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_SERVE_LMHEAD_INT4_CALIB=${HEADF:-0} E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$6" --arena "$7" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/$1_ppl_$2.json > run_$1_ppl_$2.log 2>&1
  grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" run_$1_ppl_$2.log | tail -2 | sed "s/^/    /"; }
arm(){ read G R E <<<"$(fenv $6)"; say "arm $1/$2 (B=$3 exp=$4 calib=$5 fuse=$6 head=${HEADF:-0})"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$7" E4B_SERVE_EXP_INT4=$4 E4B_SERVE_ATTN_INT4_CALIB=$5 E4B_SERVE_LMHEAD_INT4_CALIB=${HEADF:-0} E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$7" --arena "$8" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $3 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/$1_$2.json > run_$1_$2.log 2>&1
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|REFUSED|Error" run_$1_$2.log | tail -2 | sed "s/^/    /"; }
say "=============== $TAG per-fusion arms (integration-2) ==============="
for spec in "nf4 0 0 0" "int4exp 1 0 0" "calib 1 1 0" "r1 0 0 r1" "r12 0 0 r12" "epi 0 0 epi" "stack 1 0 r1epi" "all 1 1 all"; do set -- $spec; k8 $TAG $1 $2 $3 $4 "$MID" $AR; done
for spec in "nf4 0 0 0" "int4exp 1 0 0" "calib 1 1 0" "r1 0 0 r1" "r12 0 0 r12" "epi 0 0 epi" "stack 1 0 r1epi" "all 1 1 all"; do set -- $spec; arm $TAG b1_$1 1 $2 $3 $4 "$MID" $AR; done
for spec in "nf4 0 0 0" "int4exp 1 0 0" "stack 1 0 r1epi" "all 1 1 all"; do set -- $spec; arm $TAG b16_$1 16 $2 $3 $4 "$MID" $AR; done
say "=============== $TAG output-head arms (integration-3) ==============="
export PYTHONPATH=$W/hook_v2
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@claude/buildout-integration-3" > pip_bo3d.log 2>&1 || say "integration-3 install FAILED (head arms will refuse)"
python -c "from experts4bit_qlora.engines.int4_attn_calib import _LM_HEAD_ENV; print('bo3d tripwire OK: head lane present')" || say "TRIPWIRE FAIL: head lane missing"
HEADF=1 k8 $TAG head 0 0 0 "$MID" $AR;   HEADF=1 k8 $TAG allhead 1 1 all "$MID" $AR
HEADF=1 arm $TAG b1_head 1 0 0 0 "$MID" $AR;  HEADF=1 arm $TAG b1_allhead 1 1 1 all "$MID" $AR
python - <<'PY'
import json, glob, os
print("GEMMA4 SUMMARY")
for f in sorted(glob.glob("/root/bo3/gemma4_*.json")):
    try:
        d=json.load(open(f)); s=d.get("step_ms_clean"); a=d.get("aggregate_tok_s"); n=d.get("mean_nll")
        print("  %-28s %s" % (os.path.basename(f), ("%.1f tok/s" % (1000/s)) if s else (("B16 %.1f" % a) if a else ("nll %.5f" % n if n else "?"))))
    except Exception as e: print("  ", f, "unreadable", e)
PY
grep -ahE "patched|enabled|calibrated|RuntimeError|TypeError|folded|REFUSED" run_${TAG}_*.log 2>/dev/null | sort | uniq -c | sed "s/^/TPLINE $TAG /"
say "BO3D_DONE"; touch BO3D_DONE
