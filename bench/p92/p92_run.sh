#!/bin/bash
# bench/p92/p92_run.sh -- lane P92, BOX side (bench/p92/PREREG-p92.md; e4b#564). Started detached by p92_drive.sh with
# the run's nonce; P92_RUN_NONCE first, then P92_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P92_SUCCESS.<nonce> only when the reducer ran.
#
# Does K25 (grouped-nf4-gemm #429; E4B_NF4_GROUPED_SMALLM, #827) make the NF4 families' B=16 decode faster on ONE RTX
# 5090 without moving their K8? Granite-3.1-3B-A800M (r12epi) and OLMoE-1B-7B (nf4), each on its licensed env
# (bench/p44/serve_stack.py arm_env, verbatim), OFF (E4B_NF4_GROUPED_SMALLM=0, today's route) vs ON (=1: K25 for every
# NF4 decode row, T == 1 included, so K8's B=1 eager loop reads the kernel).
#   premise  on THIS card, before anything is fetched: e4b's tests/test_k25_row_exact_gpu.py (a token's K25 rows are
#            bit-equal alone and inside B=16; rc 25) and grouped-nf4-gemm's K25 contract file compiled (rc 23)
#   speed    P91's harness (graph-window timing, prompt 512, 128 generated, --no-fuse-qkv): B=16 OFF/ON/OFF/ON (the
#            first draw of each censused), then B=1 OFF/ON (censused)
#   quality  P44-a's K8 arm (2048 steps, eager B=1 loop): OFF and ON on wikitext and c4val1
# The reducer applies the registered rule per family.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P92_GPU_CLASS P92_MIN_DISK_GB P92_REHEARSAL.
set -uo pipefail
W=/root/p92; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p92: $*"; }
NONCE=${P92_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P92_RUN_NONCE.tmp && mv $W/P92_RUN_NONCE.tmp $W/P92_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P92_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P92_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P92_RUN_ID P92_DEADLINE_EPOCH P92_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=8cc3510085f329763459d128b3a3b09a822c9951   # grouped-nf4-gemm main at K25's merge (#429); a registered constant
GR=ibm-granite/granite-3.1-3b-a800m-instruct; GR_REV=a02780686e08a03fe0d2679a293b5c74a90efa89
OL=allenai/OLMoE-1B-7B-0924-Instruct; OL_REV=7f1c97f440f06ce36705e4f2b843edb5925f4498
GPU_CLASS=${P92_GPU_CLASS:-5090}; MIN_DISK_GB=${P92_MIN_DISK_GB:-100}; REHEARSAL=${P92_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
# every lever is unset; each arm sets its family's licensed env (bench/p44/serve_stack.py arm_env, verbatim) + the arm
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_CALIB_SOURCE E4B_CALIB_NSEQ E4B_MODEL_ID
GR_ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
OL_ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0"
ARM_OFF="E4B_NF4_GROUPED_SMALLM=0"; ARM_ON="E4B_NF4_GROUPED_SMALLM=1"
: > summary.txt; echo "$P92_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA granite=$GR@$GR_REV olmoe=$OL@$OL_REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 100 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p92_reduce.py p42_reduce.py test_k25_row_exact_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p92/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (two checkpoints and their arenas)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P92_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P92_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P92_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; P37 pins)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, json, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
from nf4_smallm import gemm_nf4_grouped_smallm  # noqa: F401  (K25, the treatment)
from experts4bit_qlora.engines.hot_residency import _K25_PLAN, _k25_mode_env  # noqa: F401  (the route, #827)
assert _K25_PLAN == {"block_n": 32, "kc": 256, "warps": 4, "stages": 2, "lut": "pair"}, _K25_PLAN
from experts4bit_qlora.k8_gate import verdict  # noqa: F401
import usercustomize  # noqa: F401
assert usercustomize.__file__.startswith("/root/p92/hook/"), usercustomize.__file__
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p92/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
# the reducer's self-test runs on the INSTALLED experts4bit_qlora.k8_gate, the rule the reduction applies
python $W/p92_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched. (1) e4b's route: a token's K25 rows are bit-equal alone
# (T == 1) and inside a B=16 step -- without it the B=1 K8 does not stand for the B=16 rows (rc 25). (2) K25's own
# contract compiled on this card's architecture (sm_120 has never run it; the A2000 is sm_86) (rc 23).
(cd $W && PYTHONPATH= perl -e 'alarm 900; exec @ARGV' python -m pytest test_k25_row_exact_gpu.py -q -p no:cacheprovider) > logs/rowexact.log 2>&1
rc=$?; { echo -n "premise row-exact rc=$rc: "; tail -1 logs/rowexact.log; } | tee -a summary.txt
[ "$rc" = 0 ] || { say "PREMISE FAILED: K25's rows are not row-count invariant on this card (or the test errored)"; echo "premise failed rc=$rc" > REFUSAL; finish 25; }
say "clone grouped-nf4-gemm @$GNF4_SHA for K25's contract file"
perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4 > logs/clone_gnf4.log 2>&1 && git -C $W/gnf4 checkout -q $GNF4_SHA \
  || { say "clone failed"; finish 9; }
(cd $W/gnf4/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_nf4_grouped_smallm_interp.py -q -p no:cacheprovider) > logs/k25_contract.log 2>&1
rc=$?; { echo -n "premise K25 contract compiled rc=$rc: "; tail -1 logs/k25_contract.log; } | tee -a summary.txt
[ "$rc" = 0 ] || { say "PREMISE FAILED: K25's contract does not hold on this card"; echo "k25 contract failed rc=$rc" > REFUSAL; finish 23; }
# ---- the arms
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# speed TAG MID QA ENVS ARM(off|on) B DRAW CENSUS(0|1)
speed(){ local TAG=$1 MID=$2 QA=$3 ENVS=$4 ARM=$5 B=$6 DRAW=$7 CEN=$8 AE AL extra=(); AL=$(arm_alarm 1500)
  [ "$ARM" = on ] && AE=$ARM_ON || AE=$ARM_OFF
  local name=${TAG}_${ARM}_b${B}_d$DRAW
  [ "$CEN" = 1 ] && extra=(--replay-profile-out $W/logs/census_$name.txt)
  say "speed $name (alarm=$AL)"
  env $ENVS $AE E4B_MODEL_ID=$MID perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
    --placement-override all-vram --amort off --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
    "${extra[@]}" --out $W/e4b_$name.json > logs/run_$name.log 2>&1
  local rc=$?
  { echo -n "speed $name rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_$name.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  return $rc; }
# k8 TAG MID QA ENVS ARM(off|on) SRC -- P44-a's k8_arm in flags (2048 steps, eager B=1 loop, no-fuse-qkv)
k8(){ local TAG=$1 MID=$2 QA=$3 ENVS=$4 ARM=$5 SRC=$6 AE AL; AL=$(arm_alarm 1500)
  [ "$ARM" = on ] && AE=$ARM_ON || AE=$ARM_OFF
  local name=${TAG}_k8_${ARM}_$SRC
  say "k8 $name (alarm=$AL)"
  env $ENVS $AE E4B_MODEL_ID=$MID perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
    --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv \
    --ppl-source $SRC --out $W/$name.json > logs/run_$name.log 2>&1
  local rc=$?
  { echo -n "k8 $name rc=$rc "; grep -aE "K8_PPL" logs/run_$name.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  return $rc; }
family(){ local TAG=$1 MID=$2 REV=$3 ENVS=$4
  can_run 1500 "${TAG}_fetch" || return 40
  say "fetch $MID @ $REV"
  perl -e "alarm $(arm_alarm 1500); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch_$TAG.log 2>&1 || { tail -2 logs/fetch_$TAG.log; say "DL FAIL $TAG"; return 11; }
  say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e "alarm $(arm_alarm 1500); exec @ARGV" python $W/k8_bake.py > logs/bake_$TAG.log 2>&1 || { tail -3 logs/bake_$TAG.log; say "BAKE FAIL $TAG"; return 12; }
  local QA=$W/work_$TAG/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL $TAG: no arena"; return 12; }
  # the registered order: B=16 ABAB (draw 1 censused), B=1 AB (censused), K8 OFF/ON per text
  for D in 1 2; do for ARM in off on; do
    can_run 420 "${TAG}_${ARM}_b16_d$D" && { speed $TAG "$MID" "$QA" "$ENVS" $ARM 16 $D $([ $D = 1 ] && echo 1 || echo 0); rec $?; }
  done; done
  for ARM in off on; do can_run 420 "${TAG}_${ARM}_b1_d1" && { speed $TAG "$MID" "$QA" "$ENVS" $ARM 1 1 1; rec $?; }; done
  for SRC in wikitext c4val1; do for ARM in off on; do
    can_run 600 "${TAG}_k8_${ARM}_$SRC" && { k8 $TAG "$MID" "$QA" "$ENVS" $ARM $SRC; rec $?; }
  done; done
  return 0; }
# the registered order: Granite (r12epi), then OLMoE (nf4)
family granite "$GR" "$GR_REV" "$GR_ENV"; rec $?
family olmoe "$OL" "$OL_REV" "$OL_ENV"; rec $?
say "reduce"; python $W/p92_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
