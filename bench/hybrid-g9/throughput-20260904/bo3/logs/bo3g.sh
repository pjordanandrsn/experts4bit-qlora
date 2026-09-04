#!/bin/bash
# bo3g: behind bo3b -- gpt-oss with the NATIVE MXFP4 expert store (e4b#372, integration-2 @0119cde): K8 int4exp/stack,
# B=1 int4exp/stack, B=16 int4exp/stack. The gate: K8 vs NF4 6.33544 against the 0.0176-nat floor.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3g: $*"; }
say "waiting for BO3F_DONE"; while [ ! -f BO3F_DONE ]; do sleep 30; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v2
say "install: e4b @0119cde (integration-4 = -3 + unfused rotary fold)"
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@0119cde" > pip_bo3g.log 2>&1 || { say "pip FAILED"; tail -3 pip_bo3c.log; touch BO3G_DONE; exit 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; touch BO3G_DONE; exit 9; }
from experts4bit_qlora.engines.int4_experts import _mxfp4_store_layout
from mxfp4_grouped import gemm_mxfp4_grouped
import experts4bit_qlora as e; print("bo3g tripwire OK: unfused rotary fold; e4b", e.__version__)
PYT
export PYTHONPATH=$W/hook_v2
python -c "from experts4bit_qlora.engines.glue_r2 import _patch_attention_unfused; print('bo3g tripwire OK: unfused rotary fold present')" || { say "TRIPWIRE FAIL"; touch BO3G_DONE; exit 9; }
fenv(){ case "$1" in 0) echo "0 0 0";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8g(){ # $1 tag $2 arm $3 model $4 arena   (int4 + calib + all fusions; the fold engages on unfused attention)
  read G R E <<<"$(fenv all)"; say "K8 $1/$2"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$3" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$3" --arena "$4" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/$1_ppl_$2.json > run_$1_ppl_$2.log 2>&1
  grep -aE "K8_PPL|patched|folded|REFUSED|Error" run_$1_ppl_$2.log | tail -2 | sed "s/^/    ROPE /"
}
armg(){ read G R E <<<"$(fenv all)"; say "arm $1/$2 (B=1 all fusions)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$3" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$3" --arena "$4" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/$1_$2.json > run_$1_$2.log 2>&1
  grep -aE "B1D_TIMED|patched|REFUSED|Error" run_$1_$2.log | tail -2 | sed "s/^/    ROPE /"
}
# Qwen3: the full stack with the unfused rotary fold engaged (vs bo3's all = 156.1 tok/s)
k8g qwen3 all_rope Qwen/Qwen3-30B-A3B $W/work_qwen3/nf4.arena; armg qwen3 b1_all_rope Qwen/Qwen3-30B-A3B $W/work_qwen3/nf4.arena
# Gemma-4: its round-2 fold refuses by design, so its best LICENSED stack is int4 + calib + r1 + epilogue (no r2)
GM=google/gemma-4-26B-A4B-it; GA=$W/work_gemma4/nf4.arena
say "K8 gemma4/best (int4 + calib + r1 + epilogue)"
env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$GM" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=1 \
  perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$GM" --arena "$GA" --calib $W/calib.json --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/gemma4_ppl_best.json > run_gemma4_ppl_best.log 2>&1
grep -aE "K8_PPL|REFUSED|Error" run_gemma4_ppl_best.log | tail -2 | sed "s/^/    ROPE /"
for B in 1 16; do
  say "arm gemma4/b${B}_best (int4 + calib + r1 + epilogue)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$GM" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=1 \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$GM" --arena "$GA" --calib $W/calib.json --placement-override all-vram --amort off --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/gemma4_b${B}_best.json > run_gemma4_b${B}_best.log 2>&1
  grep -aE "B1D_TIMED|BV3_|REFUSED|Error" run_gemma4_b${B}_best.log | tail -2 | sed "s/^/    ROPE /"
done
python - <<'PY2'
import json, glob, os
print("ROPE SUMMARY")
for f in sorted(glob.glob("/root/bo3/*_all_rope.json") + glob.glob("/root/bo3/*_b1_all_rope.json") + glob.glob("/root/bo3/*_ppl_all.json") + glob.glob("/root/bo3/*_b1_all.json")):
    try:
        d=json.load(open(f)); s=d.get("step_ms_clean"); n=d.get("mean_nll")
        print("  %-28s %s" % (os.path.basename(f), ("%.1f tok/s (%.2f ms)" % (1000/s, s)) if s else ("nll %.5f" % n if n else "?")))
    except Exception as e: print("  ", f, "unreadable", e)
PY2
say "BO3G_DONE"; touch BO3G_DONE
