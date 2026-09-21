#!/bin/bash
# bench/p57/p57_run.sh -- lane P57, BOX side (bench/p57/P57-PREREG.md). Started detached by p57_drive.sh with the
# run's nonce; P57_RUN_NONCE first, then P57_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P57_SUCCESS.<nonce> only on clean completion.
#
# P54's runner (P42's census protocol) with P57's arms on one RTX 5090: (a) K17's P4 -- the int4 stack with
# grouped-nf4-gemm's fused split-K reduce OFF vs ON (GNF4_GEMV_FUSED_REDUCE=0/1) at B=1 (on P54's licensed
# fused-qkv stack) and at B=16; (b) P54's P5 split -- fused qkv vs control with the round-2 glue forced OFF on
# both legs (E4B_FUSE_T1_GLUE_R2=0); every timed arm drawn TWICE interleaved (A B A B), census on the first
# draw only; then (c) one UNTIMED arm that counts the DISTINCT experts each layer routes per decode step
# (bench/p57/distinct_experts.py through the P57 hook; E4B_FUSE_ROUTER_EPI=0 because the fused epilogue never
# calls the router module). Nothing here changes a default.
set -uo pipefail
W=/root/p57; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p57: $*"; }
NONCE=${P57_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P57_RUN_NONCE.tmp && mv $W/P57_RUN_NONCE.tmp $W/P57_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P57_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P57_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P57_RUN_ID P57_DEADLINE_EPOCH P57_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; export E4B_MODEL_ID=$MID
: > summary.txt; echo "$P57_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py distinct_experts.py staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p57/staged.sha256"; finish 9; }
# ---- deadline guard: never start an arm that cannot finish 10 min before the launcher tears the box down
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P57_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P57_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
# ---- per-arm alarm: what is LEFT before the launcher's deadline, minus a 10-min fetch margin, not a
# literal. p39-box1b-4 died on a hardcoded 5400 s: that host's streamed calibration had not finished its
# FIRST chunk in 88 min where a good host does all five in 43 (100% GPU util at 107 W on a 5090 -- a
# latency-bound host, not an OOM), so the run burned 90 min and VOIDed with 4 h of paid wallclock unused.
# An arm now gets the time the run actually has; a slow host either finishes or is host-limited AT the
# deadline, which is a fact about the host rather than about a constant nobody re-read.
arm_alarm(){ local left=$(( P57_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 1800 ] && left=1800; [ "$left" -gt 18000 ] && left=18000; echo "$left"; }
# A deadline-derived alarm still lets a bad host spend the WHOLE rental before saying so. box1b-4's host
# had not finished calibration chunk 1 in 88 min where the previous host did all five in 43. So a
# calibrated arm is killed early if chunk 1 does not land within P57_FIRST_CHUNK_S (default 1500 s, ~3x
# the 8.6 min a good host takes): that bounds a bad host to ~$0.3 instead of a full 6-hour rental, and
# reports host-limited with the evidence. Arms that never calibrate pass straight through.
first_chunk_watchdog(){ local pid=$1 name=$2 budget=${P57_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)
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
# ---- install: e4b pinned + P37's toolchain pins; gnf4 at the K16 cut
export DEBIAN_FRONTEND=noninteractive
say "install e4b @$E4B_SHA (image python; P37 pins)"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
GNF4_LIVE=K16; GNF4_LIVE_SHA=$GNF4_SHA; PLAN_HAS_R=$(python -c "import inspect, int4_b32; print('R' in inspect.signature(int4_b32._plan).parameters)") || finish 9
echo "GNF4 K16 $GNF4_SHA PLAN_HAS_R=$PLAN_HAS_R" >> summary.txt
PYTHONPATH=$W/hook python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
sig = inspect.signature(f).parameters
assert "assignment" in sig and "dump_artifact_dir" in sig and "expected_fingerprint" in sig, "e4b cut lacks #405/#531 knobs"
from experts4bit_qlora.engines.pack_manifest import ASSIGNMENT_PATH, read_assignment  # noqa: F401  (#531)
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16, "e4b cut lacks the K16 route (#578)"
_smallm_kernels()   # the installed gnf4 must carry int4_smallm, or the route would refuse at enable time
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p57/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
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
hdr(){ { echo "P57 arm=$1 gnf4=${GNF4_LIVE:-?} sha=${GNF4_LIVE_SHA:-?} PLAN_HAS_R=${PLAN_HAS_R:-?} at=$(date -u +%FT%TZ)"; } > logs/run_$1.log; }
# speed_arm NAME B EXP CA [env...]   receipt e4b_b${B}_${NAME}.json ; fuse=all (1 1 1) like P37's licensed arms; nf4 control fuse=0
speed_arm(){ local NAME=$1 B=$2 EXP=$3 CA=$4; shift 4; local G=1 R=1 E=1; [ "$EXP" = 0 ] && { G=0; R=0; E=0; }
  local TAG=$NAME${P57_ARM_SUFFIX:-}   # the second draw of an arm writes <arm>_r2 receipts/logs (prereg: A B A B)
  local sp t_arm; t_arm=$(date +%s); sp=$(vram_start $TAG); hdr $TAG
  env E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E "$@" \
    perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv ${P57_EXTRA_FLAGS:-} \
      ${P57_CENSUS_OUT:+--replay-profile-out "$P57_CENSUS_OUT"} \
      --out $W/e4b_b${B}_$TAG.json >> logs/run_$TAG.log 2>&1 &
  local pid=$! rc=0
  first_chunk_watchdog "$pid" "$TAG" || rc=$?
  local wrc=0; wait "$pid" 2>/dev/null || wrc=$?
  # `[ "$rc" = 0 ] && rc=$?` captured the TEST's status, not wait's, so EVERY arm reported rc=0:
  # p39-box3's honoured32 raised a RuntimeError and the lane walked past it and wrote an empty
  # artifact line. An exit code read from the wrong command is worse than no exit code.
  [ "$rc" = 0 ] && rc=$wrc
  vram_stop $sp
  grep -aE "B1D_TIMED|BV3_|REPLAY_PROFILE_OUT|INT4EXP|ATTNINT4|REFUSED|Error" logs/run_$TAG.log | tail -4 | sed "s/^/    /"
  local nch; nch=$(grep -ac "INT4EXP calibrated experts" logs/run_$TAG.log 2>/dev/null); nch=${nch:-0}
  [ "$nch" -gt 0 ] && echo "CHUNKS $TAG $nch calibration chunk(s) in $(( $(date +%s) - t_arm ))s" >> summary.txt
  # ATTNINT4's projection count is the cross-check that the uncalibrated swap covers the
  # same modules the calibrated one did: P39's logs read "192 projections" on this model.
  grep -aoE "ATTNINT4 rtn: [0-9]+ projections" logs/run_$TAG.log | tail -1 | sed "s/^/ATTNINT4 $TAG /" >> summary.txt
  { echo -n "arm $TAG B=$B gnf4=${GNF4_LIVE:-?} rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_$TAG.log | tail -1 | cut -c1-240; echo; } >> summary.txt; return $rc; }
# k8_arm NAME ARMKIND(nf4|all) SRC [env...]  -- p37c's k8() verbatim in flags; receipt qwen3_ppl_${NAME}_${SRC}.json
k8_arm(){ local NAME=$1 KIND=$2 SRC=$3; shift 3; local EXP=1 CA=1 G=1 R=1 E=1; [ "$KIND" = nf4 ] && { EXP=0; CA=0; G=0; R=0; E=0; }
  hdr ${NAME}_$SRC
  env E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E "$@" \
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

# ---- P57 arms. Every timed arm censuses on its FIRST draw. Int4 experts (RTN) + uncalibrated int4 attention on
# every arm (P42 amendment 1); K16 route at its auto default. GNF4_GEMV_FUSED_REDUCE is read by the installed
# grouped-nf4-gemm's gemv_int4_b32 at call time (K17): 0 = the shipped two-launch path, 1 = the fused epilogue.
# fqkv arms pass --fuse-qkv AFTER speed_arm's --no-fuse-qkv (argparse: last wins). R2 legs pass
# E4B_FUSE_T1_GLUE_R2=0 explicitly: speed_arm puts the caller's "$@" LAST on its env line and `env` takes the
# last assignment, so a caller override wins (P54's speed_arm had "$@" FIRST -- an override there would have lost).
arm(){ local NAME=$1 B=$2 SUFFIX=$3 CENSUS=$4 FLAGS=$5; shift 5; local out=$W/logs/census_$NAME.txt
  [ "$CENSUS" = 1 ] || out=""
  P57_CENSUS_OUT=$out P57_ARM_SUFFIX=$SUFFIX P57_EXTRA_FLAGS=$FLAGS speed_arm "$NAME" "$B" 1 0 E4B_SERVE_ATTN_INT4=1 "$@"
}
rc_any=0
# (a) K17 P4 -- A B A B at B=1 on the fused-qkv stack, then A B A B at B=16 on the unfused stack
arm int4_b1        1  ""  1 --fuse-qkv GNF4_GEMV_FUSED_REDUCE=0 || rc_any=$?
arm int4_b1_fr     1  ""  1 --fuse-qkv GNF4_GEMV_FUSED_REDUCE=1 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b1        1  _r2 0 --fuse-qkv GNF4_GEMV_FUSED_REDUCE=0 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b1_fr     1  _r2 0 --fuse-qkv GNF4_GEMV_FUSED_REDUCE=1 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b16       16 ""  1 ""         GNF4_GEMV_FUSED_REDUCE=0 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b16_fr    16 ""  1 ""         GNF4_GEMV_FUSED_REDUCE=1 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b16       16 _r2 0 ""         GNF4_GEMV_FUSED_REDUCE=0 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b16_fr    16 _r2 0 ""         GNF4_GEMV_FUSED_REDUCE=1 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
# (b) P54's P5 split -- round-2 glue OFF on both legs; A B A B at B=16
arm int4_b16_nor2      16 ""  1 ""         GNF4_GEMV_FUSED_REDUCE=0 E4B_FUSE_T1_GLUE_R2=0 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b16_fqkv_nor2 16 ""  1 --fuse-qkv GNF4_GEMV_FUSED_REDUCE=0 E4B_FUSE_T1_GLUE_R2=0 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b16_nor2      16 _r2 0 ""         GNF4_GEMV_FUSED_REDUCE=0 E4B_FUSE_T1_GLUE_R2=0 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
arm int4_b16_fqkv_nor2 16 _r2 0 --fuse-qkv GNF4_GEMV_FUSED_REDUCE=0 E4B_FUSE_T1_GLUE_R2=0 || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }
# proof the K17 route engaged where asked and nowhere else: the fused-reduce arms' censuses carry NO
# _reduce_partials row; the control arms do (STOP-3 shape, rc 43/44)
for pair in "int4_b16:int4_b16_fr" "int4_b1:int4_b1_fr"; do ctl=${pair%%:*}; fr=${pair##*:}
  if [ -s "$W/logs/census_$fr.txt" ] && [ -s "$W/logs/census_$ctl.txt" ]; then
    n_fr=$(grep -ac "_reduce_partials" "$W/logs/census_$fr.txt"); n_ctl=$(grep -ac "_reduce_partials" "$W/logs/census_$ctl.txt")
    echo "K17ROUTE $fr reduce_rows=$n_fr control_reduce_rows=$n_ctl" >> summary.txt
    [ "$n_fr" = 0 ] || { echo "K17ROUTE NOT ENGAGED in $fr (_reduce_partials still launched)" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=43; }
    [ "$n_ctl" -gt 0 ] || { echo "K17ROUTE LEAKED into $ctl (no _reduce_partials row in the control)" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=44; }
  fi
done
for n in int4_b1 int4_b1_fr int4_b16 int4_b16_fr int4_b16_nor2 int4_b16_fqkv_nor2; do
  if [ -s "$W/logs/census_$n.txt" ]; then echo "CENSUS $n $(wc -l < $W/logs/census_$n.txt) rows" >> summary.txt
  else echo "CENSUS $n MISSING" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=40; fi
done
# (c) the distinct-expert count, UNTIMED and LAST: --amort off (the captured B>1 stage refuses --amort on, P54's
# lesson), the P57 hook attaches the router counter after enable_hybrid_tier and dumps P57_DISTINCT_OUT at exit;
# E4B_FUSE_ROUTER_EPI=0 so the router module is actually called. Its step time is recorded but never quoted.
if can_run 900 int4_b16_distinct; then
  sp=$(vram_start int4_b16_distinct); hdr int4_b16_distinct
  env E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=0 \
    P57_DISTINCT_OUT=$W/distinct_experts_b16.json P57_BATCH=16 GNF4_GEMV_FUSED_REDUCE=0 \
    perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 16 --prompt-len 512 --gen-tokens 64 --b1d-loop graph --b1d-timed --no-fuse-qkv \
      --out $W/e4b_b16_int4_b16_distinct.json >> logs/run_int4_b16_distinct.log 2>&1
  rc_d=$?; vram_stop $sp
  grep -aE "P57 HOOK|P57_DISTINCT_OUT|B1D_TIMED|BV3_|REFUSED|Error" logs/run_int4_b16_distinct.log | tail -4 | sed "s/^/    /"
  { echo -n "arm int4_b16_distinct B=16 UNTIMED rc=$rc_d "; grep -aE "P57_DISTINCT_OUT" logs/run_int4_b16_distinct.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  [ -s "$W/distinct_experts_b16.json" ] || { echo "DISTINCT MISSING" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=45; }
fi
say "----- summary -----"; cat summary.txt
finish "$rc_any"
