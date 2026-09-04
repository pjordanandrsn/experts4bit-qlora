#!/bin/bash
# bo3q: behind bo3p -- MXFP4 route probe v3: the store path against the NF4 PATH on identical inputs (E4B_INT4_KEEP_NF4=1 keeps
# the stacks). probe2 showed every row of the store path equals an in-situ dequant of the store's own bytes; this asks whether
# those bytes compute what the trusted NF4 path computes. Also the SAME-window K8 triple: NF4 / store (v1) / store (GEMV), 64 steps.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3q: $*"; }
say "waiting for BO3P_DONE"; while [ ! -f BO3P_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "install: bo3n cut (gnf4@998bbfc + e4b@c437461 = #372 + route)"
perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@998bbfcaee100f5c87dd8f027e13ee6ab7f75322" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@c437461bcd292098db405c2cc3cfe6f661aa0a53" > pip_bo3q.log 2>&1 || { say "PIP FAIL"; touch BO3Q_DONE; exit 9; }
python -c "from experts4bit_qlora.engines.int4_experts import _mxfp4_store_layout; from mxfp4_grouped import gemv_mxfp4_b32; print('bo3q tripwire OK')" || { say "TRIPWIRE FAIL"; touch BO3Q_DONE; exit 9; }
MID=openai/gpt-oss-20b; AR=$W/work_gptoss/nf4.arena
say "probe3: store (v1) vs NF4 path, KEEP_NF4=1, 16 steps"
env PYTHONPATH=$W/hook_probe3 E4B_INT4_KEEP_NF4=1 E4B_MXFP4_GEMV=0 E4B_ROUTE_PROBE=1 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
    E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0 \
  perl -e 'alarm 1800; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
    --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 16 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/gptoss_ppl_probe3.json > run_gptoss_ppl_probe3.log 2>&1
grep -a "ROUTEPROBE3\|K8_PPL\|Error" run_gptoss_ppl_probe3.log | head -16 | cut -c1-300
k8q(){ # $1 tag $2 exp $3 keep $4 gemv
  say "K8 64 steps gptoss/$1 (exp=$2 keep=$3 gemv=$4)"
  env PYTHONPATH=$W/hook_v3 E4B_INT4_KEEP_NF4=$3 E4B_MXFP4_GEMV=$4 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$2 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0 \
    perl -e 'alarm 1800; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 64 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/gptoss_ppl64_$1.json > run_gptoss_ppl64_$1.log 2>&1
  grep -a "K8_PPL\|Error" run_gptoss_ppl64_$1.log | tail -1 | sed "s/^/    K8Q $1 /"
}
k8q nf4 0 0 1; k8q store_v1 1 0 0; k8q store_gemv 1 0 1
say "BO3Q_DONE"; touch BO3Q_DONE
