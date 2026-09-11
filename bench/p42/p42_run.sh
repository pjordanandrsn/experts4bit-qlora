#!/bin/bash
# bench/p42/p42_run.sh -- lane P42, BOX side (P42-PREREG.md). Started detached by p42_drive.sh with the
# run's nonce; writes P42_RUN_NONCE first, then P42_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P42_SUCCESS.<nonce> only on clean completion.
#
# One box, one kernel (gnf4 NEW), four arms in a 2x2: {nf4, int4} x {B=1, B=16}. Every arm asks for the
# kernel census, which #555 made reachable at B>1. The lane exists to ATTRIBUTE a step time, not to
# improve one: nothing here changes a kernel.
#
# The int4 arms are deliberately UNCALIBRATED (round-to-nearest packing). The census measures which
# kernels run and for how long; gptq vs rtn changes the WEIGHTS the same kernels read, not the kernels.
# Skipping calibration removes ~20 min and most of the cost. P42-PREREG.md registers the check that
# keeps this honest: int4_b16's step time must land within 5% of p39-box4's calibrated 12.39 ms, or the
# census is of a different configuration and the lane says so instead of analysing it.
set -uo pipefail
W=/root/p42; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p42: $*"; }
NONCE=${P42_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P42_RUN_NONCE.tmp && mv $W/P42_RUN_NONCE.tmp $W/P42_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P42_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P42_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P42_RUN_ID P42_DEADLINE_EPOCH P42_INSTANCE_ID E4B_SHA GNF4_NEW_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; export E4B_MODEL_ID=$MID
: > summary.txt; echo "$P42_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p42/staged.sha256"; finish 9; }
# ---- deadline guard: never start an arm that cannot finish 10 min before the launcher tears the box down
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P42_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P42_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
# ---- per-arm alarm: what is LEFT before the launcher's deadline, minus a 10-min fetch margin, not a
# literal. p39-box1b-4 died on a hardcoded 5400 s: that host's streamed calibration had not finished its
# FIRST chunk in 88 min where a good host does all five in 43 (100% GPU util at 107 W on a 5090 -- a
# latency-bound host, not an OOM), so the run burned 90 min and VOIDed with 4 h of paid wallclock unused.
# An arm now gets the time the run actually has; a slow host either finishes or is host-limited AT the
# deadline, which is a fact about the host rather than about a constant nobody re-read.
arm_alarm(){ local left=$(( P42_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 1800 ] && left=1800; [ "$left" -gt 18000 ] && left=18000; echo "$left"; }
# A deadline-derived alarm still lets a bad host spend the WHOLE rental before saying so. box1b-4's host
# had not finished calibration chunk 1 in 88 min where the previous host did all five in 43. So a
# calibrated arm is killed early if chunk 1 does not land within P42_FIRST_CHUNK_S (default 1500 s, ~3x
# the 8.6 min a good host takes): that bounds a bad host to ~$0.3 instead of a full 6-hour rental, and
# reports host-limited with the evidence. Arms that never calibrate pass straight through.
first_chunk_watchdog(){ local pid=$1 name=$2 budget=${P42_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)
  case "$name" in *build*|honoured*|recipe*) ;; *) return 0;; esac   # only arms that calibrate
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "logs/run_$name.log" 2>/dev/null && { say "$name: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
      say "HOST-LIMITED: $name produced no calibration chunk in ${budget}s (a good host: ~520s) -- killing the arm"
      echo "HOSTLIMITED $name no calibration chunk in ${budget}s" >> summary.txt
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30
    fi
    sleep 20
  done
  return 0; }
# ---- install: e4b pinned + P37's toolchain pins; gnf4 switchable
export DEBIAN_FRONTEND=noninteractive
say "install e4b @$E4B_SHA (image python; P37 pins)"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
gnf4_switch(){ # gnf4_switch OLD|NEW  -- reinstall the kernel package at the named sha and PROVE which plan is live
  local which=$1 sha; [ "$which" = NEW ] && sha=$GNF4_NEW_SHA || sha=${GNF4_OLD_SHA:?this lane only installs NEW}
  perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$sha" > logs/pip_gnf4_$which.log 2>&1 || { tail -3 logs/pip_gnf4_$which.log; say "PIP FAIL (gnf4 $which)"; return 1; }
  GNF4_LIVE=$which; GNF4_LIVE_SHA=$sha
  PLAN_HAS_R=$(python -c "import inspect, int4_b32; print('R' in inspect.signature(int4_b32._plan).parameters)") || return 1
  local want=False; [ "$which" = NEW ] && want=True
  [ "$PLAN_HAS_R" = "$want" ] || { say "TRIPWIRE: gnf4 $which installed but PLAN_HAS_R=$PLAN_HAS_R (want $want)"; return 1; }
  say "gnf4 $which @$sha live: PLAN_HAS_R=$PLAN_HAS_R"; echo "GNF4 $which $sha PLAN_HAS_R=$PLAN_HAS_R" >> summary.txt; }
gnf4_switch NEW || finish 9
PYTHONPATH=$W/hook python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
sig = inspect.signature(f).parameters
assert "assignment" in sig and "dump_artifact_dir" in sig and "expected_fingerprint" in sig, "e4b cut lacks #405/#531 knobs"
from experts4bit_qlora.engines.pack_manifest import ASSIGNMENT_PATH, read_assignment  # noqa: F401  (#531)
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p42/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK: e4b", e.__version__, "torch", torch.__version__, "triton", triton.__version__)
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
# power at load separates a slow host from a slow kernel: box1b-4 sat at 100% util / 107 W on a 5090
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt; lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
# ---- fetch (pinned), bake (bo7's k8_bake.py), prompts (step_decomp's own window) -- as P37
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; import sys; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
say "bake NF4 arena"; mkdir -p $W/work_qwen3
K8_MODEL="$MID" K8_WORK="$W/work_qwen3" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL (no arena)"; finish 12; }
# ---- arm helpers (P37's e4b_arm / p37c's k8, with the artifact knobs)
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
hdr(){ { echo "P39 arm=$1 gnf4=${GNF4_LIVE:-?} sha=${GNF4_LIVE_SHA:-?} PLAN_HAS_R=${PLAN_HAS_R:-?} at=$(date -u +%FT%TZ)"; } > logs/run_$1.log; }
# speed_arm NAME B EXP CA [env...]   receipt e4b_b${B}_${NAME}.json ; fuse=all (1 1 1) like P37's licensed arms; nf4 control fuse=0
speed_arm(){ local NAME=$1 B=$2 EXP=$3 CA=$4; shift 4; local G=1 R=1 E=1; [ "$EXP" = 0 ] && { G=0; R=0; E=0; }
  local sp t_arm; t_arm=$(date +%s); sp=$(vram_start $NAME); hdr $NAME
  env "$@" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
      ${P42_CENSUS_OUT:+--replay-profile-out "$P42_CENSUS_OUT"} \
      --out $W/e4b_b${B}_$NAME.json >> logs/run_$NAME.log 2>&1 &
  local pid=$! rc=0
  first_chunk_watchdog "$pid" "$NAME" || rc=$?
  local wrc=0; wait "$pid" 2>/dev/null || wrc=$?
  # `[ "$rc" = 0 ] && rc=$?` captured the TEST's status, not wait's, so EVERY arm reported rc=0:
  # p39-box3's honoured32 raised a RuntimeError and the lane walked past it and wrote an empty
  # artifact line. An exit code read from the wrong command is worse than no exit code.
  [ "$rc" = 0 ] && rc=$wrc
  vram_stop $sp
  grep -aE "B1D_TIMED|BV3_|REPLAY_PROFILE_OUT|INT4EXP|ATTNINT4|REFUSED|Error" logs/run_$NAME.log | tail -4 | sed "s/^/    /"
  local nch; nch=$(grep -ac "INT4EXP calibrated experts" logs/run_$NAME.log 2>/dev/null); nch=${nch:-0}
  [ "$nch" -gt 0 ] && echo "CHUNKS $NAME $nch calibration chunk(s) in $(( $(date +%s) - t_arm ))s" >> summary.txt
  # ATTNINT4's projection count is the cross-check that the uncalibrated swap covers the
  # same modules the calibrated one did: P39's logs read "192 projections" on this model.
  grep -aoE "ATTNINT4 rtn: [0-9]+ projections" logs/run_$NAME.log | tail -1 | sed "s/^/ATTNINT4 $NAME /" >> summary.txt
  { echo -n "arm $NAME B=$B gnf4=${GNF4_LIVE:-?} rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_$NAME.log | tail -1 | cut -c1-240; echo; } >> summary.txt; return $rc; }
# k8_arm NAME ARMKIND(nf4|all) SRC [env...]  -- p37c's k8() verbatim in flags; receipt qwen3_ppl_${NAME}_${SRC}.json
k8_arm(){ local NAME=$1 KIND=$2 SRC=$3; shift 3; local EXP=1 CA=1 G=1 R=1 E=1; [ "$KIND" = nf4 ] && { EXP=0; CA=0; G=0; R=0; E=0; }
  hdr ${NAME}_$SRC
  env "$@" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source $SRC --out $W/qwen3_ppl_${NAME}_$SRC.json >> logs/run_${NAME}_$SRC.log 2>&1 &
  local pid=$! rc=0
  # box 2's calibrated arms run HERE, not through speed_arm: the watchdog was added to speed_arm only,
  # so p39-box2-2's honoured build sat 1h51m on a latency-bound host (machine 36493, 100% util at 108 W,
  # zero chunks) with the deadline-derived alarm giving it five hours. A guard that covers one of two
  # call sites is not a guard.
  first_chunk_watchdog "$pid" "${NAME}_$SRC" || rc=$?
  local wrc=0; wait "$pid" 2>/dev/null || wrc=$?
  [ "$rc" = 0 ] && rc=$wrc
  grep -aE "K8_PPL|INT4EXP calibrated experts|honoured|ATTNINT4|REFUSED|Error" logs/run_${NAME}_$SRC.log | tail -4 | sed "s/^/    /"
  { echo -n "k8 $NAME src=$SRC rc=$rc "; grep -aE "K8_PPL" logs/run_${NAME}_$SRC.log | tail -1 | cut -c1-240; echo; } >> summary.txt; return $rc; }
fp_of(){ python -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }

# ---- the 2x2. Every arm censuses; --replay-profile-out runs 8 profiled replays AFTER the timed
# window, in slots cap_tokens already reserves, so the timed number stays byte-identical to a
# no-flag run (step_decomp's own contract, e4b#275).
# Amendment 1. The int4 arms take UNCALIBRATED int4 attention (E4B_SERVE_ATTN_INT4=1),
# never the calibrated flag. enable_serve_attn_int4 and enable_serve_attn_int4_calib
# install the same Int4Linear from int4_attn.py and differ only in the packer, so the
# serve-time kernels -- what a census measures -- are identical, while the calibrated
# path additionally runs a Hessian pass over 192 projections. That pass is a BUILD cost
# and P39 paid it once, in its build arm, then loaded the result from an artifact. This
# lane has no artifact, so every int4 arm was paying it fresh: run 1 (p42-census-2) spent
# 40 minutes inside it on int4_b1 and never reached a timed window. Same reasoning as the
# experts, which the prereg already applied and this line failed to.
census_arm(){ local NAME=$1 B=$2 EXP=$3; local extra=()
  [ "$EXP" = 1 ] && extra=(E4B_SERVE_ATTN_INT4=1)
  # ${extra[@]+...}: an empty array under `set -u` is an unbound expansion on bash < 4.4
  P42_CENSUS_OUT=$W/logs/census_$NAME.txt speed_arm "$NAME" "$B" "$EXP" 0 ${extra[@]+"${extra[@]}"}
}
rc_any=0
can_run 900 nf4_b1    && { census_arm nf4_b1    1  0 || rc_any=$?; }
can_run 900 nf4_b16   && { census_arm nf4_b16   16 0 || rc_any=$?; }
can_run 900 int4_b1   && { census_arm int4_b1   1  1 || rc_any=$?; }
can_run 900 int4_b16  && { census_arm int4_b16  16 1 || rc_any=$?; }
for n in nf4_b1 nf4_b16 int4_b1 int4_b16; do
  if [ -s "$W/logs/census_$n.txt" ]; then
    echo "CENSUS $n $(wc -l < $W/logs/census_$n.txt) rows" >> summary.txt
  else
    echo "CENSUS $n MISSING" >> summary.txt; rc_any=${rc_any:-40}; [ "$rc_any" = 0 ] && rc_any=40
  fi
done
say "----- summary -----"; cat summary.txt
finish "$rc_any"
