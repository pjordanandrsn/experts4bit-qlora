#!/bin/bash
# bo3f: the optimisation pass's first step -- B=1 kernel censuses (8 untimed graph replays after the timed window) per
# family at NF4 and at its best LICENSED stack, so each family's largest slice is measured, not guessed.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3f: $*"; }
say "waiting for BO3E_DONE and the route probe"; while [ ! -f BO3E_DONE ]; do sleep 30; done
while ! grep -q "mxprobe3: done" mxprobe3.log 2>/dev/null; do sleep 15; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v2
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
cen(){ # $1 tag $2 name $3 exp $4 calib $5 fuse $6 head $7 model $8 arena
  read G R E <<<"$(fenv $5)"; say "census $1/$2 (exp=$3 calib=$4 fuse=$5 head=$6)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$7" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_SERVE_LMHEAD_INT4_CALIB=$6 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$7" --arena "$8" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
      --replay-profile-out $W/census_$1_$2.txt --out $W/$1_cen_$2.json > run_census_$1_$2.log 2>&1
  grep -aE "B1D_TIMED|REPLAY_PROFILE|Error|error" run_census_$1_$2.log | tail -2 | sed "s/^/    CENSUS /"
}
# gpt-oss (baked): NF4 and its licensed stack (r1 + epilogue)
cen gptoss nf4   0 0 0     0 openai/gpt-oss-20b $W/work_gptoss/nf4.arena
cen gptoss stack 0 0 r1epi 0 openai/gpt-oss-20b $W/work_gptoss/nf4.arena
# Gemma-4 (baked): NF4, r1, and int4 + r1 + epilogue
cen gemma4 nf4   0 0 0     0 google/gemma-4-26B-A4B-it $W/work_gemma4/nf4.arena
cen gemma4 r1    0 0 r1    0 google/gemma-4-26B-A4B-it $W/work_gemma4/nf4.arena
cen gemma4 stack 1 0 r1epi 0 google/gemma-4-26B-A4B-it $W/work_gemma4/nf4.arena
# Qwen3 (baked, the reference): NF4 and all
cen qwen3 nf4    0 0 0     0 Qwen/Qwen3-30B-A3B $W/work_qwen3/nf4.arena
cen qwen3 all    1 1 all   0 Qwen/Qwen3-30B-A3B $W/work_qwen3/nf4.arena
# Granite: re-download (6 GB) + bake, then NF4 and its licensed stack
MID=ibm-granite/granite-3.1-3b-a800m-instruct
python - "$MID" <<'PYD' > dl_granite_redo.log 2>&1
import os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
for a in range(4):
    try:
        print("OK", snapshot_download(sys.argv[1], allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken"], ignore_patterns=["original/*","consolidated*"], max_workers=4)); break
    except Exception as e:
        print("retry", a, repr(e)[:120], flush=True); time.sleep(20)
PYD
if grep -q "^OK " dl_granite_redo.log; then
  K8_MODEL="$MID" K8_WORK="$W/work_granite" perl -e 'alarm 3600; exec @ARGV' python $W/k8_bake.py > bake_granite_redo.log 2>&1
  if python -c "import json,sys;d=json.load(open('$W/work_granite/bake.json'));sys.exit(0 if d.get('status')=='OK' else 1)" 2>/dev/null; then
    cen granite nf4   0 0 0     0 "$MID" $W/work_granite/nf4.arena
    cen granite stack 1 0 r1epi 0 "$MID" $W/work_granite/nf4.arena
  else say "granite bake failed"; fi
else say "granite download failed: $(tail -1 dl_granite_redo.log | cut -c1-100)"; fi
ls -la $W/census_*.txt | awk '{print "CENSUSFILE", $5, $9}'
say "BO3F_DONE"; touch BO3F_DONE
