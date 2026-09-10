#!/bin/bash
# bench/p39/p39_run.sh -- lane P39, BOX side (P39-PREREG.md). Started detached by p39_drive.sh with the run's
# nonce; writes P39_RUN_NONCE first, then P39_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P39_SUCCESS.<nonce> only on clean completion. bo7's step_decomp.py / k8_bake.py / calib.json + hook v7 are
# staged next to this file and sha-checked against staged.sha256 before anything runs.
#   P39_BOX=1  H1: one artifact-pinned pack, gnf4 OLD/NEW alternating (kernel A/B), controls, dump for box 2
#   P39_BOX=2  H2: honour box 1's assignment.json, dump, K8 two-text gate on the honoured pack vs NF4
set -uo pipefail
W=/root/p39; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p39/box${P39_BOX:-?}: $*"; }
NONCE=${P39_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P39_RUN_NONCE.tmp && mv $W/P39_RUN_NONCE.tmp $W/P39_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P39_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P39_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P39_BOX P39_RUN_ID P39_DEADLINE_EPOCH P39_INSTANCE_ID E4B_SHA GNF4_NEW_SHA GNF4_OLD_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; export E4B_MODEL_ID=$MID
: > summary.txt; echo "$P39_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p39/staged.sha256"; finish 9; }
# ---- deadline guard: never start an arm that cannot finish 10 min before the launcher tears the box down
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P39_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P39_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
# ---- per-arm alarm: what is LEFT before the launcher's deadline, minus a 10-min fetch margin, not a
# literal. p39-box1b-4 died on a hardcoded 5400 s: that host's streamed calibration had not finished its
# FIRST chunk in 88 min where a good host does all five in 43 (100% GPU util at 107 W on a 5090 -- a
# latency-bound host, not an OOM), so the run burned 90 min and VOIDed with 4 h of paid wallclock unused.
# An arm now gets the time the run actually has; a slow host either finishes or is host-limited AT the
# deadline, which is a fact about the host rather than about a constant nobody re-read.
arm_alarm(){ local left=$(( P39_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 1800 ] && left=1800; [ "$left" -gt 18000 ] && left=18000; echo "$left"; }
# A deadline-derived alarm still lets a bad host spend the WHOLE rental before saying so. box1b-4's host
# had not finished calibration chunk 1 in 88 min where the previous host did all five in 43. So a
# calibrated arm is killed early if chunk 1 does not land within P39_FIRST_CHUNK_S (default 1500 s, ~3x
# the 8.6 min a good host takes): that bounds a bad host to ~$0.3 instead of a full 6-hour rental, and
# reports host-limited with the evidence. Arms that never calibrate pass straight through.
first_chunk_watchdog(){ local pid=$1 name=$2 budget=${P39_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)
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
  local which=$1 sha; [ "$which" = NEW ] && sha=$GNF4_NEW_SHA || sha=$GNF4_OLD_SHA
  perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$sha" > logs/pip_gnf4_$which.log 2>&1 || { tail -3 logs/pip_gnf4_$which.log; say "PIP FAIL (gnf4 $which)"; return 1; }
  GNF4_LIVE=$which; GNF4_LIVE_SHA=$sha
  PLAN_HAS_R=$(python -c "import inspect, int4_b32; print('R' in inspect.signature(int4_b32._plan).parameters)") || return 1
  local want=False; [ "$which" = NEW ] && want=True
  [ "$PLAN_HAS_R" = "$want" ] || { say "TRIPWIRE: gnf4 $which installed but PLAN_HAS_R=$PLAN_HAS_R (want $want)"; return 1; }
  say "gnf4 $which @$sha live: PLAN_HAS_R=$PLAN_HAS_R"; echo "GNF4 $which $sha PLAN_HAS_R=$PLAN_HAS_R" >> summary.txt; }
[ "$P39_BOX" = 1 ] && FIRST=OLD || FIRST=NEW
gnf4_switch $FIRST || finish 9
PYTHONPATH=$W/hook python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
sig = inspect.signature(f).parameters
assert "assignment" in sig and "dump_artifact_dir" in sig and "expected_fingerprint" in sig, "e4b cut lacks #405/#531 knobs"
from experts4bit_qlora.engines.pack_manifest import ASSIGNMENT_PATH, read_assignment  # noqa: F401  (#531)
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p39/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
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
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv --out $W/e4b_b${B}_$NAME.json >> logs/run_$NAME.log 2>&1 &
  local pid=$! rc=0
  first_chunk_watchdog "$pid" "$NAME" || rc=$?
  wait "$pid" 2>/dev/null; [ "$rc" = 0 ] && rc=$?
  vram_stop $sp
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|gptq /|honoured|REFUSED|Error" logs/run_$NAME.log | tail -4 | sed "s/^/    /"
  local nch; nch=$(grep -ac "INT4EXP calibrated experts" logs/run_$NAME.log 2>/dev/null || echo 0)
  [ "$nch" -gt 0 ] && echo "CHUNKS $NAME $nch calibration chunk(s) in $(( $(date +%s) - t_arm ))s" >> summary.txt
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
  wait "$pid" 2>/dev/null; [ "$rc" = 0 ] && rc=$?
  grep -aE "K8_PPL|INT4EXP calibrated experts|honoured|ATTNINT4|REFUSED|Error" logs/run_${NAME}_$SRC.log | tail -4 | sed "s/^/    /"
  { echo -n "k8 $NAME src=$SRC rc=$rc "; grep -aE "K8_PPL" logs/run_${NAME}_$SRC.log | tail -1 | cut -c1-240; echo; } >> summary.txt; return $rc; }
fp_of(){ python -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }
CAL="E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128"
if [ "$P39_BOX" = 1 ]; then
  # ---------------- BOX 1: controls, one calibration + dump, then the kernel A/B on identical bytes
  if [ "${P39_BUILD_ONLY:-0}" != 1 ]; then
    can_run 900 nf4_b1  && speed_arm nf4_b1 1 0 0
    can_run 900 nf4_b16 && speed_arm nf4_b16 16 0 0
  fi
  can_run 3600 old_b16_build || finish 20
  speed_arm old_b16_build 16 1 1 $CAL E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact || { brc=$?
    [ "$brc" = 30 ] && { say "host-limited on the build; nothing measured about the hypothesis"; finish 30; }
    say "licensed build failed (rc=$brc)"; finish 20; }
  FP=$(fp_of $W/artifact); [ -n "$FP" ] || { say "no artifact fingerprint after the build"; finish 20; }
  mkdir -p $W/box1_out && cp $W/artifact/manifest.json $W/artifact/payloads/assignment.json $W/artifact/payloads/identity.json $W/box1_out/ && echo "$FP" > $W/box1_out/FINGERPRINT
  say "artifact $FP dumped; assignment staged for box 2 in box1_out/"; echo "ARTIFACT $FP" >> summary.txt
  # Amendment 3: the assignment must name EVERY expert layer, or the record cannot drive a re-pack
  NL=$(python -c "import json; print(len({r['layer'] for r in json.load(open('$W/box1_out/assignment.json'))['method_map']}))")
  echo "ASSIGNMENT_LAYERS $NL" | tee -a summary.txt
  [ "$NL" = 48 ] || { say "VOID: assignment names $NL of 48 layers"; finish 21; }
  if [ "${P39_BUILD_ONLY:-0}" = 1 ]; then say "build-only run (Amendment 3): dump complete, no A/B arms"; grep -a "PLAN_HAS_R\|gptq /\|ARTIFACT\|ASSIGNMENT" summary.txt logs/run_*.log 2>/dev/null | sort -u | head -40 > highlights.txt; echo "----- summary -----"; cat summary.txt; finish 0; fi
  LOAD="E4B_SERVE_EXP_INT4_CALIB=1 E4B_INT4_ARTIFACT_DIR=$W/artifact E4B_INT4_EXPECTED_FINGERPRINT=$FP"
  # ABAB on identical bytes; each arm's log header carries PLAN_HAS_R, the per-arm proof of which kernel ran
  can_run 700 old_b16_r1 && speed_arm old_b16_r1 16 1 1 $LOAD
  gnf4_switch NEW || finish 9
  can_run 700 new_b16_r1 && speed_arm new_b16_r1 16 1 1 $LOAD
  can_run 700 new_b1_r1  && speed_arm new_b1_r1  1  1 1 $LOAD
  gnf4_switch OLD || finish 9
  can_run 700 old_b1_r1  && speed_arm old_b1_r1  1  1 1 $LOAD
  can_run 700 old_b16_r2 && speed_arm old_b16_r2 16 1 1 $LOAD
  gnf4_switch NEW || finish 9
  can_run 700 new_b16_r2 && speed_arm new_b16_r2 16 1 1 $LOAD
else
  # ---------------- BOX 2: honour box 1's decision, gate it
  [ -s $W/box1/assignment.json ] && [ -n "${P39_BOX1_FINGERPRINT:-}" ] || { say "box 2 needs box1/assignment.json and P39_BOX1_FINGERPRINT"; finish 78; }
  can_run 600 nf4_wikitext && k8_arm nf4 nf4 wikitext
  can_run 600 nf4_c4val1   && k8_arm nf4 nf4 c4val1
  can_run 3600 honoured_build || finish 20
  k8_arm honoured all wikitext $CAL E4B_INT4_ASSIGNMENT=$W/box1/assignment.json E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact2 || { hrc=$?
    [ "$hrc" = 30 ] && { say "host-limited on the honoured build; H2 not measured"; finish 30; }
    say "honoured build failed (rc=$hrc)"; finish 20; }
  FP2=$(fp_of $W/artifact2); echo "ARTIFACT2 $FP2 BOX1 $P39_BOX1_FINGERPRINT $([ "$FP2" = "$P39_BOX1_FINGERPRINT" ] && echo SAME_BYTES || echo DIFFERENT_BYTES)" | tee -a summary.txt
  mkdir -p $W/box2_out && cp $W/artifact2/manifest.json $W/artifact2/payloads/assignment.json $W/box2_out/ 2>/dev/null
  can_run 700 honoured_c4 && k8_arm honoured all c4val1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_INT4_ARTIFACT_DIR=$W/artifact2 E4B_INT4_EXPECTED_FINGERPRINT=$FP2
  python - <<'PYV' | tee -a summary.txt
import json, os
W="/root/p39"
def ppl(a, s):
    p=f"{W}/qwen3_ppl_{a}_{s}.json"
    if not os.path.exists(p): return None
    d=json.load(open(p)); return next((float(d[k]) for k in ("ppl","k8_ppl","perplexity") if k in d), None)
out={"rule":"ppl(honoured)-ppl(nf4) <= +0.05 on wikitext AND c4val1","texts":{}}; ok=True
for s in ("wikitext","c4val1"):
    n, a = ppl("nf4", s), ppl("honoured", s); d = (a-n) if (a is not None and n is not None) else None
    v = "PASS" if d is not None and d <= 0.05 else ("MISSING" if d is None else "FAIL"); ok = ok and v=="PASS"
    out["texts"][s]={"nf4":n,"honoured":a,"delta":d,"verdict":v}
out["verdict"]="PASS (split was the cause)" if ok else "FAIL or MISSING (split not the cause, or host-limited)"
json.dump(out, open(f"{W}/gate_verdict.json","w"), indent=1); print("GATE", json.dumps(out))
PYV
  # optional: plain recipe on this box -- do its counts differ from box 1's? (P37's flip, third box)
  can_run 3600 recipe_build && k8_arm recipe all wikitext $CAL E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact3 && echo "ARTIFACT3 $(fp_of $W/artifact3)" >> summary.txt
fi
grep -a "PLAN_HAS_R\|gptq /\|honoured\|ARTIFACT\|GATE\|SKIPPED" summary.txt logs/run_*.log 2>/dev/null | grep -v "^logs.*P39 arm" | sort -u | head -40 > highlights.txt
echo "----- summary -----"; cat summary.txt
finish 0
