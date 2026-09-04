#!/bin/bash
# bo3l: behind bo3k -- the OP-LEVEL census (e4b#380 --op-profile-out) on the two stacks whose kernel census is glue-heavy:
# Qwen3 `all` (torch elementwise 22% + reduce 13% + topk/index 7% of 5.56 ms) and Granite `stackr2` (elementwise 28% of 2.85 ms).
# Two eager steps after a short timed window, profiled with stacks + shapes: names the ops and call sites to fuse next.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3l: $*"; }
say "waiting for BO3K_DONE"; while [ ! -f BO3K_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v3
python -c "import experts4bit_qlora as e, int4_b32; print('bo3l cut: e4b', e.__version__, 'rope_heads', hasattr(int4_b32, 'rope_heads'))"
opp(){ # $1 tag $2 arm $3 exp $4 calib $5 G $6 R $7 E $8 model $9 arena
  say "op-profile $1/$2 (exp=$3 calib=$4 fuse=$5$6$7)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$8" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$5 E4B_FUSE_T1_GLUE_R2=$6 E4B_FUSE_ROUTER_EPI=$7 \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp_op.py --model "$8" --arena "$9" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 32 --b1d-loop eager --b1d-timed --no-fuse-qkv \
      --op-profile-out $W/opcensus_$1_$2.txt --out $W/$1_op_$2.json > run_op_$1_$2.log 2>&1
  grep -aE "B1D_TIMED|OP_PROFILE_OUT|Traceback|Error" run_op_$1_$2.log | tail -2 | sed "s/^/    OPCENSUS /"
  [ -f $W/opcensus_$1_$2.txt ] && { echo "    OPCENSUS $1/$2 top ops by count:"; sed -n '/== by op (launch count) ==/,/== by op + input shape/p' $W/opcensus_$1_$2.txt | grep -E "^\s*aten::|^\s*cuda|^\s*_" | head -14 | awk '{print "    OPCENSUS   " $1, $NF}'; }
}
opp qwen3 all 1 1 1 1 1 Qwen/Qwen3-30B-A3B $W/work_qwen3/nf4.arena
opp granite stackr2 1 0 1 1 1 ibm-granite/granite-3.1-3b-a800m-instruct $W/work_granite/nf4.arena
say "BO3L_DONE"; touch BO3L_DONE
