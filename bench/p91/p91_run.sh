#!/bin/bash
# bench/p91/p91_run.sh -- lane P91, BOX side (bench/p91/PREREG-p91.md; e4b#564). Started detached by p91_drive.sh with
# the run's nonce; P91_RUN_NONCE first, then P91_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P91_SUCCESS.<nonce> only when the reducer ran.
#
# Where do the NF4 families' decode steps go? A descriptive kernel census -- no change is tested, nothing is licensed --
# of Granite-3.1-3B-A800M and OLMoE-1B-7B on their LICENSED configurations (bo7: Granite r12epi = NF4 experts + folds +
# router epilogue; OLMoE nf4), at B=16 and B=1 on ONE RTX 5090, with the current defaults (K19 / lean glue / K21 do not
# touch NF4 stores). P86's/P88's harness (bench/p39/step_decomp.py + bench/p42 hook), graph-window timing, P42 replay
# census, one draw per arm. The reducer reconciles each census against its step and ranks the kernels; its decision rule
# (pre-registered) names the next lane.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P91_GPU_CLASS P91_MIN_DISK_GB P91_REHEARSAL.
set -uo pipefail
W=/root/p91; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p91: $*"; }
NONCE=${P91_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P91_RUN_NONCE.tmp && mv $W/P91_RUN_NONCE.tmp $W/P91_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P91_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P91_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P91_RUN_ID P91_DEADLINE_EPOCH P91_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=4cc831cfa55076f5fe6a23991b88b1c10ca38d87   # grouped-nf4-gemm main (P90's pin); a registered constant
GR=ibm-granite/granite-3.1-3b-a800m-instruct; GR_REV=a02780686e08a03fe0d2679a293b5c74a90efa89
OL=allenai/OLMoE-1B-7B-0924-Instruct; OL_REV=7f1c97f440f06ce36705e4f2b843edb5925f4498
GPU_CLASS=${P91_GPU_CLASS:-5090}; MIN_DISK_GB=${P91_MIN_DISK_GB:-100}; REHEARSAL=${P91_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
# every lever is unset; each arm sets its family's licensed env (bench/p44/serve_stack.py arm_env, verbatim)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_CALIB_SOURCE E4B_CALIB_NSEQ E4B_MODEL_ID
GR_ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
OL_ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0"
: > summary.txt; echo "$P91_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA granite=$GR@$GR_REV olmoe=$OL@$OL_REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 100 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p91_reduce.py p42_reduce.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p91/staged.sha256"; finish 9; }
python $W/p91_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (two checkpoints and their arenas)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P91_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P91_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P91_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
import nf4_grouped  # noqa: F401  (the NF4 grouped GEMM this census sizes)
import usercustomize  # noqa: F401
assert usercustomize.__file__.startswith("/root/p91/hook/"), usercustomize.__file__
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p91/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
# family TAG MID REV ENV -- fetch (pinned), bake (P39's k8_bake.py), then the census arms at B=16 and B=1
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
census(){ local TAG=$1 MID=$2 QA=$3 ENVS=$4 B=$5 AL; AL=$(arm_alarm 1500)
  say "census $TAG B=$B (alarm=$AL)"
  env $ENVS E4B_MODEL_ID=$MID perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
    --placement-override all-vram --amort off --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
    --replay-profile-out $W/logs/census_${TAG}_b$B.txt --out $W/e4b_${TAG}_b$B.json > logs/run_${TAG}_b$B.log 2>&1
  local rc=$?
  { echo -n "census $TAG B=$B rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_${TAG}_b$B.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  return $rc; }
family(){ local TAG=$1 MID=$2 REV=$3 ENVS=$4
  can_run 1800 "${TAG}_fetch" || return 40
  say "fetch $MID @ $REV"
  perl -e "alarm $(arm_alarm 1500); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch_$TAG.log 2>&1 || { tail -2 logs/fetch_$TAG.log; say "DL FAIL $TAG"; return 11; }
  say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e "alarm $(arm_alarm 1500); exec @ARGV" python $W/k8_bake.py > logs/bake_$TAG.log 2>&1 || { tail -3 logs/bake_$TAG.log; say "BAKE FAIL $TAG"; return 12; }
  local QA=$W/work_$TAG/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL $TAG: no arena"; return 12; }
  for B in 16 1; do can_run 600 "${TAG}_b$B" && { census $TAG "$MID" "$QA" "$ENVS" $B; rec $?; }; done
  return 0; }
# the registered order: Granite (r12epi), then OLMoE (nf4)
family granite "$GR" "$GR_REV" "$GR_ENV"; rec $?
family olmoe "$OL" "$OL_REV" "$OL_ENV"; rec $?
say "reduce"; python $W/p91_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
