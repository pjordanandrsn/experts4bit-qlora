#!/bin/bash
# bo3p: behind bo3o -- op census REDO on the merged mains (bo3l's Qwen3 run died on a hook/cut mismatch, its Granite run attributed every op to the tracer's own frame).
# Cut: e4b@af53396bbeda6899b5cf7192bd09993629719b27 (main + site fix) + gnf4@a213b88c5b4adf872e5f2617625a90db97733058 (main = 0.28.0 content). Qwen3 `all` (int4 + calib + r12 + epi) and Granite's LICENSED stack nf4_r12epi.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3p: $*"; }
say "waiting for BO3O_DONE"; while [ ! -f BO3O_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v3
say "install: e4b@af53396bbeda6899b5cf7192bd09993629719b27 + gnf4@a213b88c5b4adf872e5f2617625a90db97733058"
perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@a213b88c5b4adf872e5f2617625a90db97733058" "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@af53396bbeda6899b5cf7192bd09993629719b27" > pip_bo3p.log 2>&1 || { say "PIP FAIL"; tail -3 pip_bo3p.log; touch BO3P_DONE; exit 9; }
python -c "
from int4_b32 import rope_heads; from experts4bit_qlora.engines.glue_r2 import _patch_attention_rope_only, _patch_attention_unfused
from experts4bit_qlora.engines.int4_attn_calib import _dense_mlp_linears; import experts4bit_qlora as e; print('bo3p tripwire OK', e.__version__)" || { say "TRIPWIRE FAIL"; touch BO3P_DONE; exit 9; }
opp(){ # $1 tag $2 arm $3 exp $4 calib $5 G $6 R $7 E $8 model $9 arena
  say "op-profile $1/$2 (exp=$3 calib=$4 fuse=$5$6$7)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$8" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$5 E4B_FUSE_T1_GLUE_R2=$6 E4B_FUSE_ROUTER_EPI=$7 \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp_op.py --model "$8" --arena "$9" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 32 --b1d-loop eager --b1d-timed --no-fuse-qkv \
      --op-profile-out $W/opcensus2_$1_$2.txt --out $W/$1_op2_$2.json > run_op2_$1_$2.log 2>&1
  grep -aE "B1D_TIMED|OP_PROFILE_OUT|Traceback|Error" run_op2_$1_$2.log | tail -2 | sed "s/^/    OPCENSUS2 /"
  [ -f $W/opcensus2_$1_$2.txt ] && { echo "    OPCENSUS2 $1/$2 top sites by device time:"; sed -n '5,16p' $W/opcensus2_$1_$2.txt | cut -c1-230 | sed "s/^/    OPCENSUS2   /"; }
}
opp qwen3 all 1 1 1 1 1 Qwen/Qwen3-30B-A3B $W/work_qwen3/nf4.arena
opp granite nf4_r12epi 0 0 1 1 1 ibm-granite/granite-3.1-3b-a800m-instruct $W/work_granite/nf4.arena
say "BO3P_DONE"; touch BO3P_DONE
