#!/bin/bash
# bench/p88/p88_run.sh -- lane P88, BOX side (bench/p88/PREREG-p88.md; e4b#564). Started detached by p88_drive.sh with
# the run's nonce; P88_RUN_NONCE first, then P88_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P88_SUCCESS.<nonce> only when the reducer ran.
#
# P87's instrument, re-run on K19's new default plan (gnf4 K20, BLOCK_N 32 / KC 256: 0.736x the served route on
# recorded routing, bit-identical outputs). Does it make the int4 stack's decode faster on an RTX 5090 without moving
# its quality? Two changes from P87, both from its VOID read: the host's CPU vendor is floored (the calibration build is
# CPU-bound; P87's Broadwell host ran out of its alarm), and the build's alarm and the guard are longer. On ONE box:
#   speed    B=16 and B=1, the opt-in OFF (the split-K GEMV) and ON (K19), each drawn twice, the first draw censused.
#            P58/P86's harness (bench/p39/step_decomp.py + bench/p42 hook) and P86's int4 env (RTN int4 experts,
#            uncalibrated int4 attention, glue r1/r2, router epilogue) and command line (graph replay, fused q/k/v).
#   quality  K8 (wikitext, 2,048 teacher-forced steps at B=1, window 9ef10d760ad9) on the LICENSED recipe: calibrated
#            experts + calibrated int4 attention, P85's env and K8 arguments. One build calibrates the experts and dumps
#            the pack (opt-in OFF); then the pack is loaded by fingerprint for OFF and ON.
#   premise  tests/test_k19_row_exact_gpu.py on this card: a token's K19 rows are bit-equal alone (T == 1) and inside
#            a B=16 step, so K8 (a T == 1 instrument) stands for the batched rows. Run before anything is fetched.
# The reducer applies the pre-registered rule.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P88_GPU_CLASS P88_CPU_VENDOR P88_MIN_DISK_GB P88_CALIB_NSEQ P88_REHEARSAL.
# P88_PROVE=1 is the PROVING RUN: the refusals, the install with its tripwire, the reducer self-test, the premise test,
# grouped-nf4-gemm's K19 + K16 contract tests compiled on this card (K19 has never run on sm_120), an egress probe;
# no model.
set -uo pipefail
W=/root/p88; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p88: $*"; }
NONCE=${P88_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P88_RUN_NONCE.tmp && mv $W/P88_RUN_NONCE.tmp $W/P88_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P88_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P88_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P88_RUN_ID P88_DEADLINE_EPOCH P88_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=7b7e6b15c8628586727b037b201f4d252016ad20   # grouped-nf4-gemm at K20's read (#421): K19's default plan BLOCK_N 32 / KC 256; a registered constant
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
GPU_CLASS=${P88_GPU_CLASS:-5090}; CPU_VENDOR=${P88_CPU_VENDOR:-AuthenticAMD}; MIN_DISK_GB=${P88_MIN_DISK_GB:-200}; NSEQ=${P88_CALIB_NSEQ:-128}; REHEARSAL=${P88_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
# every lever is unset; each arm sets its own switches (E4B_INT4_GROUPED_SMALLM explicitly, OFF 0 or ON 1)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB \
      E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_INT4_ARTIFACT_DIR E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_DUMP_ARTIFACT_DIR E4B_INT4_ASSIGNMENT \
      E4B_CALIB_NSEQ E4B_CALIB_SOURCE E4B_CALIB_LAYERS_PER_PASS E4B_INT4_GROUPED_SMALLM
: > summary.txt; echo "$P88_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MID rev=$REV gpu_class=$GPU_CLASS cpu_vendor=$CPU_VENDOR min_disk_gb=$MIN_DISK_GB calib_nseq=$NSEQ" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$CPU_VENDOR" != AuthenticAMD ] || [ "$MIN_DISK_GB" != 200 ] || [ "$NSEQ" != 128 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p88_reduce.py p42_reduce.py test_k19_row_exact_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p88/staged.sha256"; finish 9; }
python $W/p88_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
# the host CPU floor (P87's VOID): the calibrated K8 build is CPU-bound, and P85's licensed floats were read on AMD hosts
# (P84 amendment 1). rc 18 is a host-floor refusal the launcher excludes the machine on (adertha#131).
HOST_VENDOR=$(lscpu | awk -F: '/^Vendor ID/{gsub(/ /, "", $2); print $2; exit}')
echo "cpu_vendor $HOST_VENDOR" | tee -a forensics.txt
[ "$HOST_VENDOR" = "$CPU_VENDOR" ] || { say "REFUSED: host CPU vendor is '${HOST_VENDOR:-unknown}', the lane registers $CPU_VENDOR (the calibration build is CPU-bound; P87)"; echo "refused: cpu vendor ${HOST_VENDOR:-unknown}" > REFUSAL; finish 18; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the bf16 checkpoint, an NF4 arena and the expert pack)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P88_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P88_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P88_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
# one pip install with ONE retry after 20 s: a git clone can fail transiently (P83's A2000 rehearsal: git exit 128)
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
# ---- install: e4b at the launch commit + P37's toolchain pins (image python); gnf4 at K19's merge
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; P37 pins)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, json, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
from int4_smallm import gemm_int4_b32_grouped_smallm  # noqa: F401  (K19 is installed)
from experts4bit_qlora.engines import hot_residency as hr
assert "E4B_INT4_GROUPED_SMALLM" in inspect.getsource(hr), "e4b lacks the K19 route (#803)"
assert hasattr(hr, "_collapsed_grouping"), "e4b lacks the T == 1 extension (#804): K8 would read the GEMV"
_p = inspect.signature(gemm_int4_b32_grouped_smallm).parameters
assert (_p["block_n"].default, _p["kc"].default) == (32, 256), "gnf4 lacks K20's default plan (BLOCK_N 32 / KC 256)"
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16 and hasattr(Int4Linear, "fuse"), "e4b cut lacks the K16 route or Int4Linear.fuse"
_smallm_kernels()
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
for k in ("artifact_dir", "expected_fingerprint", "dump_artifact_dir"):
    assert k in inspect.signature(f).parameters, f"e4b cut lacks the #405 knob {k!r}"
import int4_b32
assert int4_b32.gemv_fused_reduce_default() is False, "the fused reduce is defaulted on (the OFF arms read the two-launch default)"
import usercustomize  # noqa: F401
assert usercustomize.__file__.startswith("/root/p88/hook/"), usercustomize.__file__
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p88/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "K19 + both routes present")
PYT
cat versions.txt | tee -a summary.txt
# ---- the premise, on THIS card: K19's rows are bit-equal alone (T == 1) and inside a B=16 step. Without it a B=1 K8
# does not stand for the B=16 rows, so the lane stops before anything is fetched (rc 25).
(cd $W && PYTHONPATH= perl -e 'alarm 900; exec @ARGV' python -m pytest test_k19_row_exact_gpu.py -q -p no:cacheprovider) > logs/rowexact.log 2>&1
rc=$?; { echo -n "PREMISE row-exact rc=$rc: "; tail -1 logs/rowexact.log; } | tee -a summary.txt
python -c "import json; json.dump({'rc': $rc, 'last': open('$W/logs/rowexact.log').read().strip().splitlines()[-1:]}, open('$W/rowexact.json', 'w'))"
[ "$rc" = 0 ] || { say "PREMISE FAILED: K19's rows are not row-count invariant on this card (or the test errored)"; echo "premise failed rc=$rc" > REFUSAL; finish 25; }
if [ "${P88_PROVE:-0}" = 1 ]; then
  echo "PROVE -- the proving run: install tripwired; premise held; K19 + K16 contract tests compiled on this card" | tee -a summary.txt
  say "PROVE: clone grouped-nf4-gemm @$GNF4_SHA for its contract tests"
  perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4 > logs/clone_gnf4.log 2>&1 && git -C $W/gnf4 checkout -q $GNF4_SHA \
    || { say "PROVE: clone failed"; finish 9; }
  (cd $W/gnf4/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_int4_grouped_smallm_interp.py test_int4_smallm_interp.py -q -p no:cacheprovider) > logs/k19_contract.log 2>&1
  rc=$?; { echo -n "PROVE K19+K16 contract compiled rc=$rc: "; tail -1 logs/k19_contract.log; } | tee -a summary.txt
  [ "$rc" = 0 ] || { say "PROVE: the K19 contract does not hold on this card"; finish 23; }
  say "PROVE: HF CDN egress probe (50 MB range, 20 s cap; recorded, not a refusal)"
  python - <<'PYE' 2>&1 | tail -1 | tee -a summary.txt forensics.txt
import time, urllib.request
t = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request("https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors",
                                                      headers={"Range": "bytes=0-52428799"}), timeout=20)
    n = len(r.read())
    print(f"PROVE hf_cdn_mbps={n / (time.time() - t) / 1e6:.1f} bytes={n}")
except Exception as e:
    print(f"PROVE hf_cdn_probe_failed {type(e).__name__}: {str(e)[:120]}")
PYE
  : > PROVED; finish 0
fi
# ---- fetch (pinned), bake (P39's k8_bake.py)
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm 4800); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
say "bake NF4 arena"; mkdir -p $W/work_qwen3
K8_MODEL="$MID" K8_WORK="$W/work_qwen3" perl -e "alarm $(arm_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL: no arena"; finish 12; }
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
# speed: P86's int4 env (RTN experts, uncalibrated int4 attention) and command line
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
# quality: P85's licensed env (calibrated experts, calibrated int4 attention) and K8 arguments
LICENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 E4B_CALIB_NSEQ=$NSEQ $FOLDS"
K8ARGS="--placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext"
# speed arm: K19=0|1, B, suffix, census(1|"")
speed(){ local K=$1 B=$2 SFX=$3 CEN=$4 TAG; TAG=$([ "$K" = 1 ] && echo on || echo off); local AL; AL=$(arm_alarm 1800)
  say "speed $TAG B=$B$SFX (census=${CEN:-no} alarm=$AL)"
  env $SPEEDENV E4B_INT4_GROUPED_SMALLM=$K \
    perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --fuse-qkv \
      ${CEN:+--replay-profile-out $W/logs/census_b${B}_$TAG.txt} --out $W/e4b_b${B}_$TAG$SFX.json > logs/run_b${B}_$TAG$SFX.log 2>&1
  local rc=$?
  { echo -n "speed $TAG B=$B$SFX rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_b${B}_$TAG$SFX.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  return $rc; }
# K8 build watchdog (P85 amendment 1): a host that cannot finish the first calibration chunk in 1500 s is host-limited
first_chunk_watchdog(){ local pid=$1 log=$2 t0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "$log" 2>/dev/null && { say "build: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge 1500 ]; then
      say "HOST-LIMITED: no calibration chunk in 1500s -- killing the build"; echo "HOSTLIMITED build no calibration chunk in 1500s" >> summary.txt
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30
    fi
    sleep 20
  done; return 0; }
# quality arm: NAME, watch|nowatch, then the arm's own env
k8(){ local NAME=$1 WATCH=$2; shift 2; local AL; AL=$(arm_alarm 5400)
  say "K8 $NAME (alarm=$AL)"
  env $LICENV "$@" perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json $K8ARGS \
    --out $W/k8_$NAME.json > logs/k8_$NAME.log 2>&1 &
  local pid=$! brc=0 wrc=0
  if [ "$WATCH" = watch ]; then first_chunk_watchdog "$pid" logs/k8_$NAME.log || brc=$?; fi
  wait "$pid" 2>/dev/null || wrc=$?; [ "$brc" = 0 ] && brc=$wrc
  { echo -n "K8 $NAME rc=$brc "; grep -aE "K8_PPL|REFUSED|Error" logs/k8_$NAME.log | tail -1 | cut -c1-200; echo; } | tee -a summary.txt
  [ "$brc" = 30 ] && finish 30
  return $brc; }
fp_of(){ python -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }
verifies(){ python -c "from experts4bit_qlora.engines.pack_manifest import verify_artifact as v; v('$1', expected_model_revision='$REV')" > logs/verify_$(basename $1).log 2>&1; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# the registered order: speed first draws interleaved OFF/ON (censused); the K8 build (OFF, pack dumped), K8 OFF, K8 ON
# (pack loaded by fingerprint); speed second draws ON/OFF
for B in 16 1; do
  can_run 900 b${B}_off && { speed 0 $B "" 1; rec $?; }
  can_run 900 b${B}_on && { speed 1 $B "" 1; rec $?; }
done
if can_run 5400 k8_build; then
  rm -rf $W/artifact
  k8 build watch E4B_CALIB_LAYERS_PER_PASS=10 E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact E4B_INT4_GROUPED_SMALLM=0; rec $?
  if verifies $W/artifact && FP=$(fp_of $W/artifact) && [ -n "$FP" ]; then
    mkdir -p $W/manifests && cp $W/artifact/manifest.json $W/manifests/experts.json
    echo "PACK $FP" | tee -a summary.txt
    can_run 1800 k8_off && { k8 off nowatch E4B_INT4_ARTIFACT_DIR=$W/artifact E4B_INT4_EXPECTED_FINGERPRINT=$FP E4B_INT4_GROUPED_SMALLM=0; rec $?; }
    can_run 1800 k8_on && { k8 on nowatch E4B_INT4_ARTIFACT_DIR=$W/artifact E4B_INT4_EXPECTED_FINGERPRINT=$FP E4B_INT4_GROUPED_SMALLM=1; rec $?; }
  else
    say "the build left no expert pack that verifies"; echo "BUILD FAILED (experts)" >> summary.txt; rec 20
  fi
fi
for B in 16 1; do
  can_run 900 b${B}_on_r2 && { speed 1 $B _r2 ""; rec $?; }
  can_run 900 b${B}_off_r2 && { speed 0 $B _r2 ""; rec $?; }
done
say "reduce"; python $W/p88_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
