#!/bin/bash
# bo3j: the harness's own K8 on gpt-oss with the MXFP4 store and the route probe hook armed (16 steps).
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3j: $*"; }
say "waiting for BO3I_DONE"; while [ ! -f BO3I_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_probe
MID=openai/gpt-oss-20b; AR=$W/work_gptoss/nf4.arena
say "K8 gptoss store + route probe (16 steps)"
env E4B_ROUTE_PROBE=1 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
    E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0 \
  perl -e 'alarm 1500; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
    --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 16 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/gptoss_ppl_probe.json > run_gptoss_ppl_probe.log 2>&1
grep -a "ROUTEPROBE\|K8_PPL\|Error" run_gptoss_ppl_probe.log | head -20 | cut -c1-230
say "BO3J_DONE"; touch BO3J_DONE
