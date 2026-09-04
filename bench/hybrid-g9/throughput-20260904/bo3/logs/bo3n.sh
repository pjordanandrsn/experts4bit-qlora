#!/bin/bash
# bo3n: behind bo3m -- gpt-oss decode-grade MXFP4 GEMV speed A/B (gnf4#331 gemv_mxfp4_b32 + e4b#372 route @c437461bcd292098db405c2cc3cfe6f661aa0a53).
# Arms on the NATIVE store (exp=1 = the mxfp4 store under the #372 branch): B=1 mx_int4exp with E4B_MXFP4_GEMV=1 (new GEMV) and =0 (v1 grouped GEMM),
# B=16 both, plus the NF4 baseline B=1 re-run on this cut (exp=0). Quality of the store is bo3j's question (+0.35 nats open); this phase is SPEED only.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3n: $*"; }
say "waiting for BO3M_DONE"; while [ ! -f BO3M_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v3
say "install: gnf4@998bbfcaee100f5c87dd8f027e13ee6ab7f75322 (#331) + e4b@c437461bcd292098db405c2cc3cfe6f661aa0a53 (#372 + GEMV route)"
perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@998bbfcaee100f5c87dd8f027e13ee6ab7f75322" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@c437461bcd292098db405c2cc3cfe6f661aa0a53" > pip_bo3n.log 2>&1 || { say "PIP FAIL"; tail -3 pip_bo3n.log; touch BO3N_DONE; exit 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; touch BO3N_DONE; exit 9; }
from mxfp4_grouped import gemv_mxfp4_b32, gemm_mxfp4_grouped
from experts4bit_qlora.engines.int4_experts import _mxfp4_store_layout
import experts4bit_qlora as e, inspect
from experts4bit_qlora.engines import hot_residency as h
assert "gemv_mxfp4_b32" in inspect.getsource(h), "route missing"
print("bo3n tripwire OK: gemv_mxfp4_b32 + route; e4b", e.__version__)
PYT
MID=openai/gpt-oss-20b; AR=$W/work_gptoss/nf4.arena
armn(){ # $1 arm $2 batch $3 exp $4 gemv-flag
  say "arm gptoss/$1 B=$2 (exp=$3 E4B_MXFP4_GEMV=$4)"
  env E4B_MXFP4_GEMV=$4 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0 \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $2 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/gptoss_mx2_b$2_$1.json > run_gptoss_mx2_b$2_$1.log 2>&1
  grep -aE "B1D_TIMED|BV3_|store kind|INT4EXP|REFUSED|Error" run_gptoss_mx2_b$2_$1.log | tail -2 | sed "s/^/    MXGEMV /"
}
armn nf4 1 0 1; armn gemv 1 1 1; armn v1 1 1 0; armn nf4 16 0 1; armn gemv 16 1 1; armn v1 16 1 0
for f in run_gptoss_mx2_b1_nf4 run_gptoss_mx2_b1_gemv run_gptoss_mx2_b1_v1 run_gptoss_mx2_b16_nf4 run_gptoss_mx2_b16_gemv run_gptoss_mx2_b16_v1; do grep -aE "B1D_TIMED|BV3_" $f.log | tail -1 | sed "s/^/    MXGEMV SUMMARY $f /"; done
say "BO3N_DONE"; touch BO3N_DONE
