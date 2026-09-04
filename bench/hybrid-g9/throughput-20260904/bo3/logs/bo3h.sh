#!/bin/bash
# bo3h: the harness's OWN K8 on gpt-oss with the native MXFP4 store, 512 steps, twice: NF4 stacks freed (the lane's
# configuration, +0.35 nats) vs kept (E4B_INT4_KEEP_NF4=1). Direct calls of the serve forward are exact either way,
# so this isolates whether the harness route depends on the freed stacks.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3h: $*"; }
say "waiting for BO3G_DONE"; while [ ! -f BO3G_DONE ]; do sleep 30; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v2
MID=openai/gpt-oss-20b; AR=$W/work_gptoss/nf4.arena
python -c "from experts4bit_qlora.engines.int4_experts import _mxfp4_store_layout; print('bo3h tripwire OK: mxfp4 store present')" || { say "TRIPWIRE FAIL (integration-4 lacks the store?)"; touch BO3H_DONE; exit 9; }
k8x(){ # $1 name $2 exp $3 keep
  say "K8 gptoss/$1 (exp=$2 keep_nf4=$3 steps=512)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$2 E4B_INT4_KEEP_NF4=$3 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0 \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 512 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/gptoss_ppl_mxh_$1.json > run_gptoss_ppl_mxh_$1.log 2>&1
  grep -aE "K8_PPL|INT4EXP|Error|error" run_gptoss_ppl_mxh_$1.log | tail -2 | sed "s/^/    MXH /"
}
k8x nf4 0 0
k8x store_freed 1 0
k8x store_kept 1 1
say "BO3H_DONE"; touch BO3H_DONE
