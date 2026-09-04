#!/bin/bash
# bo3k: behind bo3j -- the rotary-only fold for norm-less attention (gnf4#330 rope_heads + e4b#379), Granite A/B on one box.
# OLD cut = whatever is installed when this starts (integration-5 @cebcc2a + kernel-integration); NEW cut = gnf4@7bf3458 + e4b@b5100d9.
# Arms: stack (int4 + r1 + epi; R2 off = control, the fold cannot engage) and stackr2 (int4 + r1 + r2 + epi; the fold engages on NEW).
# Gate: K8 stackr2 NEW vs OLD inside Granite's 0.0033-nat floor; B=1 / B=16 stackr2 NEW vs OLD; census names the glue delta.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3k: $*"; }
say "waiting for BO3J_DONE"; while [ ! -f BO3J_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v3
MID=ibm-granite/granite-3.1-3b-a800m-instruct; AR=$W/work_granite/nf4.arena
fenv(){ case "$1" in r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8k(){ # $1 cut $2 arm $3 fuse-token
  read G R E <<<"$(fenv $3)"; say "K8 granite/$2 [$1]"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/granite_ppl_${2}_$1.json > run_granite_ppl_${2}_$1.log 2>&1
  grep -aE "K8_PPL|glue r2|folded|patched|REFUSED|Error" run_granite_ppl_${2}_$1.log | tail -3 | sed "s/^/    ROPEK $1 /"
}
armk(){ # $1 cut $2 arm $3 batch $4 fuse-token
  read G R E <<<"$(fenv $4)"; say "arm granite/$2 B=$3 [$1]"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $3 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/granite_b${3}_${2}_$1.json > run_granite_b${3}_${2}_$1.log 2>&1
  grep -aE "B1D_TIMED|BV3_|glue r2|folded|patched|REFUSED|Error" run_granite_b${3}_${2}_$1.log | tail -3 | sed "s/^/    ROPEK $1 /"
}
cenk(){ # $1 cut $2 arm $3 fuse-token   (B=1 kernel census over untimed graph replays)
  read G R E <<<"$(fenv $3)"; say "census granite/$2 [$1]"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
      --replay-profile-out $W/census_granite_${2}_$1.txt --out $W/granite_cen_${2}_$1.json > run_census_granite_${2}_$1.log 2>&1
  grep -aE "B1D_TIMED|REPLAY_PROFILE|Error" run_census_granite_${2}_$1.log | tail -2 | sed "s/^/    ROPEK $1 census /"
}
say "OLD cut: $(python -c 'import experts4bit_qlora as e; print(e.__version__)' 2>/dev/null) rope_heads=$(python -c 'import int4_b32; print(hasattr(int4_b32, "rope_heads"))' 2>/dev/null)"
k8k old stackr2 all; armk old stack 1 r1epi; armk old stackr2 1 all; armk old stackr2 16 all; cenk old stackr2 all
say "install NEW cut: gnf4@7bf3458 (#330 rope_heads) + e4b@b5100d9 (#379 rope-only fold)"
perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@7bf3458e4921afe6d982897e28d9aa2f34bbc8b2" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@b5100d9f42b79b3e82999b088d1c4955445c48ae" > pip_bo3k.log 2>&1 || { say "PIP FAIL"; tail -3 pip_bo3k.log; touch BO3K_DONE; exit 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; touch BO3K_DONE; exit 9; }
import int4_b32
from int4_b32 import rope_heads, rope_norm_heads, rmsnorm_resid_rows, scaled_resid_add_rows
from experts4bit_qlora.engines.glue_r2 import _patch_attention_rope_only
import experts4bit_qlora as e; print("bo3k tripwire OK: rope_heads + rope-only fold; e4b", e.__version__)
PYT
say "NEW cut: $(python -c 'import experts4bit_qlora as e; print(e.__version__)') rope_heads=$(python -c 'import int4_b32; print(hasattr(int4_b32, "rope_heads"))')"
k8k new stackr2 all; armk new stack 1 r1epi; armk new stackr2 1 all; armk new stackr2 16 all; cenk new stackr2 all
for f in granite_ppl_stackr2_old granite_ppl_stackr2_new; do grep -a "K8_PPL" run_$f.log | tail -1 | sed "s/^/    ROPEK SUMMARY $f /"; done
for f in granite_b1_stack_old granite_b1_stackr2_old granite_b16_stackr2_old granite_b1_stack_new granite_b1_stackr2_new granite_b16_stackr2_new; do grep -aE "B1D_TIMED|BV3_" run_$f.log | tail -1 | sed "s/^/    ROPEK SUMMARY $f /"; done
say "BO3K_DONE"; touch BO3K_DONE
