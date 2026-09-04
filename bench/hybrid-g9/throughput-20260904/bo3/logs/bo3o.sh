#!/bin/bash
# bo3o: behind bo3n -- MXFP4 route probe v2: EVERY row of every MoE call against the in-situ reference (bo3j checked row 0 only,
# which cannot see a tile-boundary / ragged-group defect at prefill (512 singleton groups) or a decode-row defect). Runs on bo3n's cut
# (gnf4@998bbfc + e4b@c437461): v1 grouped GEMM arm (E4B_MXFP4_GEMV=0 -- the arm that carried +0.35 nats) and the new GEMV arm (=1).
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3o: $*"; }
say "waiting for BO3N_DONE"; while [ ! -f BO3N_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_probe2
MID=openai/gpt-oss-20b; AR=$W/work_gptoss/nf4.arena
for g in 0 1; do
  say "probe2 gptoss store, E4B_MXFP4_GEMV=$g (16 steps)"
  env E4B_MXFP4_GEMV=$g E4B_ROUTE_PROBE=1 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0 \
    perl -e 'alarm 1800; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 16 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/gptoss_ppl_probe2_g$g.json > run_gptoss_ppl_probe2_g$g.log 2>&1
  grep -a "ROUTEPROBE2\|K8_PPL\|Error" run_gptoss_ppl_probe2_g$g.log | head -24 | cut -c1-260
done
say "BO3O_DONE"; touch BO3O_DONE
