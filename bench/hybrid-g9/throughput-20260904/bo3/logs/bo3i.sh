#!/bin/bash
# bo3i: behind bo3g -- integration-5 (= -4 + dense-MLP calibrated target #378 + int4 attention biases #377):
# Gemma-4 best + dense (+ head); gpt-oss with calibrated int4 attention now that biases are carried (K8 + B=1).
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3i: $*"; }
say "waiting for BO3G_DONE"; while [ ! -f BO3G_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v3
say "install: e4b @cebcc2a (integration-5)"
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@cebcc2a" > pip_bo3i.log 2>&1 || { say "pip FAILED"; tail -3 pip_bo3i.log; touch BO3I_DONE; exit 9; }
python -c "from experts4bit_qlora.engines.int4_attn_calib import _DENSE_MLP_ENV; import experts4bit_qlora.engines.int4_attn as a; import inspect; assert 'self.bias' in inspect.getsource(a.Int4Linear); from experts4bit_qlora.engines.glue_r2 import _patch_attention_unfused; print('bo3i tripwire OK: dense target, biased Int4Linear, unfused fold')" || { say "TRIPWIRE FAIL"; touch BO3I_DONE; exit 9; }
run(){ # $1 tag $2 name $3 exp $4 calib $5 dense $6 head $7 G $8 R $9 E ${10} model ${11} arena  [K8 + B=1]
  say "K8 $1/$2 (exp=$3 calib=$4 dense=$5 head=$6 fuse=$7$8$9)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="${10}" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_SERVE_DENSE_INT4_CALIB=$5 E4B_SERVE_LMHEAD_INT4_CALIB=$6 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$7 E4B_FUSE_T1_GLUE_R2=$8 E4B_FUSE_ROUTER_EPI=$9 \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "${10}" --arena "${11}" --calib $W/calib.json --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/$1_ppl_$2.json > run_$1_ppl_$2.log 2>&1
  grep -aE "K8_PPL|ATTNINT4|REFUSED|Error" run_$1_ppl_$2.log | tail -2 | sed "s/^/    DENSE /"
  say "arm $1/b1_$2"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="${10}" E4B_SERVE_EXP_INT4=$3 E4B_SERVE_ATTN_INT4_CALIB=$4 E4B_SERVE_DENSE_INT4_CALIB=$5 E4B_SERVE_LMHEAD_INT4_CALIB=$6 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$7 E4B_FUSE_T1_GLUE_R2=$8 E4B_FUSE_ROUTER_EPI=$9 \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "${10}" --arena "${11}" --calib $W/calib.json --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/$1_b1_$2.json > run_$1_b1_$2.log 2>&1
  grep -aE "B1D_TIMED|ATTNINT4|REFUSED|Error" run_$1_b1_$2.log | tail -2 | sed "s/^/    DENSE /"
}
GM=google/gemma-4-26B-A4B-it; GA=$W/work_gemma4/nf4.arena; OM=openai/gpt-oss-20b; OA=$W/work_gptoss/nf4.arena
run gemma4 bestdense     1 1 1 0 1 0 1 "$GM" $GA
run gemma4 bestdensehead 1 1 1 1 1 0 1 "$GM" $GA
run gptoss calibbias     0 1 0 0 0 0 0 "$OM" $OA
run gptoss r1epicalib    0 1 0 0 1 0 1 "$OM" $OA
run gptoss r1epicalibhead 0 1 0 1 1 0 1 "$OM" $OA
python - <<'PY'
import json, glob, os
print("DENSE SUMMARY")
for f in sorted(glob.glob("/root/bo3/gemma4_*bestdense*.json") + glob.glob("/root/bo3/gptoss_*calib*.json") + glob.glob("/root/bo3/gptoss_*r1epi*.json")):
    try:
        d=json.load(open(f)); s=d.get("step_ms_clean"); n=d.get("mean_nll")
        print("  %-32s %s" % (os.path.basename(f), ("%.1f tok/s (%.2f ms)" % (1000/s, s)) if s else ("nll %.5f" % n if n else "?")))
    except Exception as e: print("  ", f, "unreadable", e)
PY
say "BO3I_DONE"; touch BO3I_DONE
