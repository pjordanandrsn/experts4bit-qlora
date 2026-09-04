#!/bin/bash
# bo3m: behind bo3l -- Granite's LICENSED stack on NF4 experts (int4 experts FAIL the registered 0.05-ppl gate: +0.063 ppl, P32 correction).
# Arms (exp=0, calib=0): nf4_r1epi (r1 + epi) and nf4_r12epi (r1 + r2 + epi). OLD cut = whatever bo3k left? No: bo3k installs the NEW cut
# (gnf4@7bf3458 + e4b@fb20cf0) and bo3l runs on it, so this phase runs NEW first, then re-installs the OLD cut (integration-5 @cebcc2a +
# gnf4 kernel-integration) for the control. r2 on NEW engages the rope-only fold on Granite's norm-less attention; on OLD it is the scaled layer fold only.
set -uo pipefail
LANE=bo3; W=/root/$LANE; cd $W
say(){ echo "[$(date -u +%FT%TZ)] bo3m: $*"; }
say "waiting for BO3L_DONE"; while [ ! -f BO3L_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook_v3
MID=ibm-granite/granite-3.1-3b-a800m-instruct; AR=$W/work_granite/nf4.arena
fenv(){ case "$1" in r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
k8m(){ read G R E <<<"$(fenv $3)"; say "K8 granite/$2 [$1] (exp=0 calib=0 fuse=$3)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 3600; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/granite_ppl_${2}_$1.json > run_granite_ppl_${2}_$1.log 2>&1
  grep -aE "K8_PPL|REFUSED|Error" run_granite_ppl_${2}_$1.log | tail -2 | sed "s/^/    ROPEM $1 /"; }
armm(){ read G R E <<<"$(fenv $4)"; say "arm granite/$2 B=$3 [$1] (exp=0 fuse=$4)"
  env E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 2400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $3 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/granite_b${3}_${2}_$1.json > run_granite_b${3}_${2}_$1.log 2>&1
  grep -aE "B1D_TIMED|BV3_|REFUSED|Error" run_granite_b${3}_${2}_$1.log | tail -2 | sed "s/^/    ROPEM $1 /"; }
cut(){ python -c "import experts4bit_qlora as e, int4_b32; print('e4b', e.__version__, 'rope_heads', hasattr(int4_b32, 'rope_heads'))" 2>/dev/null; }
say "cut now: $(cut)"
if python -c "import int4_b32, sys; sys.exit(0 if hasattr(int4_b32, 'rope_heads') else 1)" 2>/dev/null; then
  say "NEW cut present"; k8m new nf4_r12epi all; armm new nf4_r1epi 1 r1epi; armm new nf4_r12epi 1 all; armm new nf4_r12epi 16 all
  say "install OLD cut for the control: gnf4@kernel-integration + e4b@cebcc2a"
  perl -e 'alarm 1500; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps --no-cache-dir \
    "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@claude/kernel-integration" \
    "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@cebcc2a" > pip_bo3m.log 2>&1 || { say "PIP FAIL (old cut)"; tail -3 pip_bo3m.log; }
  say "cut now: $(cut)"
  k8m old nf4_r12epi all; armm old nf4_r1epi 1 r1epi; armm old nf4_r12epi 1 all; armm old nf4_r12epi 16 all
else
  say "OLD cut present (bo3k did not install NEW?)"; k8m old nf4_r12epi all; armm old nf4_r1epi 1 r1epi; armm old nf4_r12epi 1 all; armm old nf4_r12epi 16 all
fi
for f in run_granite_ppl_nf4_r12epi_new run_granite_ppl_nf4_r12epi_old; do grep -a "K8_PPL" $f.log 2>/dev/null | tail -1 | sed "s/^/    ROPEM SUMMARY $f /"; done
for f in run_granite_b1_nf4_r1epi_new run_granite_b1_nf4_r12epi_new run_granite_b16_nf4_r12epi_new run_granite_b1_nf4_r1epi_old run_granite_b1_nf4_r12epi_old run_granite_b16_nf4_r12epi_old; do grep -aE "B1D_TIMED|BV3_" $f.log 2>/dev/null | tail -1 | sed "s/^/    ROPEM SUMMARY $f /"; done
say "BO3M_DONE"; touch BO3M_DONE
