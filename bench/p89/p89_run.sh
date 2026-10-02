#!/bin/bash
# bench/p89/p89_run.sh -- lane P89, BOX side (bench/p89/PREREG-p89.md; e4b#564). Started detached by p89_drive.sh with
# the run's nonce; P89_RUN_NONCE first, then P89_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P89_SUCCESS.<nonce> only when the reducer ran.
#
# Does K23's lean glue (E4B_INT4_LEAN_GLUE=1, #814 over grouped-nf4-gemm #427) make Qwen3-30B-A3B's B=16 int4 decode
# step faster on an RTX 5090? The route is bit-identical to the default by construction, so there is no K8 here: the
# quality gate is the on-card bit-equality premise plus generated-token equality between the arms. Derived from
# bench/p88/p88_run.sh with the K8 build and the B=1 arms removed (the lean route does not engage at T == 1, where the
# licensed default keeps the singleton GEMV). On ONE box:
#   premise  tests/test_k19_row_exact_gpu.py + tests/test_k23_lean_glue_gpu.py on this card: K19's rows are row-count
#            invariant, and the lean route is bit-equal to the default at B=16 (eager, captured, from token rows).
#            Run before anything is fetched (rc 25).
#   speed    B=16, the opt-in OFF (0) and ON (1), K19 at its licensed default (E4B_INT4_GROUPED_SMALLM unset), each
#            drawn twice, the first draw censused. P88's harness, env and command line.
# The reducer applies the pre-registered rule.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P89_GPU_CLASS P89_MIN_DISK_GB P89_REHEARSAL.
# P89_PROVE=1 is the PROVING RUN: the refusals, the install with its tripwire, the reducer self-test, the premise,
# grouped-nf4-gemm's K19 / K16 contracts and K23's builder tests compiled on this card, an egress probe; no model.
set -uo pipefail
W=/root/p89; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p89: $*"; }
NONCE=${P89_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P89_RUN_NONCE.tmp && mv $W/P89_RUN_NONCE.tmp $W/P89_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P89_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P89_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P89_RUN_ID P89_DEADLINE_EPOCH P89_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=3990dbc43ee1729a761f5ed423b503f1e09a180f   # grouped-nf4-gemm at K23's merge (#427); a registered constant
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
GPU_CLASS=${P89_GPU_CLASS:-5090}; MIN_DISK_GB=${P89_MIN_DISK_GB:-150}; REHEARSAL=${P89_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export PYTHONPATH=$W/hook E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
# every lever is unset; each arm sets its own switch (E4B_INT4_LEAN_GLUE explicitly, OFF 0 or ON 1)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB \
      E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_INT4_ARTIFACT_DIR E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_DUMP_ARTIFACT_DIR E4B_INT4_ASSIGNMENT \
      E4B_CALIB_NSEQ E4B_CALIB_SOURCE E4B_CALIB_LAYERS_PER_PASS E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE
: > summary.txt; echo "$P89_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MID rev=$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p89_reduce.py p42_reduce.py test_k19_row_exact_gpu.py test_k23_lean_glue_gpu.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p89/staged.sha256"; finish 9; }
python $W/p89_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the bf16 checkpoint and an NF4 arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P89_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P89_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P89_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
# one pip install with ONE retry after 20 s: a git clone can fail transiently (P83's A2000 rehearsal: git exit 128)
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
# ---- install: e4b at the launch commit + P37's toolchain pins (image python); gnf4 at K23's merge
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
from int4_smallm import gemm_int4_b32_grouped_smallm
from int4_b32 import build_group_tiles_fused
_p = inspect.signature(gemm_int4_b32_grouped_smallm).parameters
assert (_p["block_n"].default, _p["kc"].default) == (32, 256), "gnf4 lacks K20's default plan (BLOCK_N 32 / KC 256)"
assert {"scatter", "gather_div"} <= set(_p), "gnf4 lacks K23's K19 options (scatter=, gather_div=)"
assert {"lean", "sorted_ids"} <= set(inspect.signature(build_group_tiles_fused).parameters), "gnf4 lacks K23's builder options"
from experts4bit_qlora.engines import hot_residency as hr
assert hasattr(hr, "_lean_glue_env") and "x_tokens" in inspect.signature(hr._fused_over_stack).parameters, \
    "e4b lacks the K23 route (#814)"
assert hr._k19_mode_env() == "auto", "K19 is not at its licensed default (auto) in this environment"
assert hr._lean_glue_env() is False, "E4B_INT4_LEAN_GLUE leaked into the runner's environment"
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16 and hasattr(Int4Linear, "fuse"), "e4b cut lacks the K16 route or Int4Linear.fuse"
_smallm_kernels()
import int4_b32
assert int4_b32.gemv_fused_reduce_default() is False, "the fused reduce is defaulted on (P88's arms read the two-launch default)"
import usercustomize  # noqa: F401
assert usercustomize.__file__.startswith("/root/p89/hook/"), usercustomize.__file__
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p89/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "K19 default + K23 present")
PYT
cat versions.txt | tee -a summary.txt
# ---- the premise, on THIS card: K19's rows are row-count invariant, and the lean route is bit-equal to the default at
# B=16 (eager, captured, from token rows). Without it the arms' token equality could not stand for the route's bits,
# so the lane stops before anything is fetched (rc 25).
(cd $W && PYTHONPATH= perl -e 'alarm 900; exec @ARGV' python -m pytest test_k19_row_exact_gpu.py test_k23_lean_glue_gpu.py -q -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; { echo -n "PREMISE k19 row-exact + k23 bit-equal rc=$rc: "; tail -1 logs/premise.log; } | tee -a summary.txt
python -c "import json; json.dump({'rc': $rc, 'last': open('$W/logs/premise.log').read().strip().splitlines()[-1:]}, open('$W/premise.json', 'w'))"
[ "$rc" = 0 ] || { say "PREMISE FAILED: K19's rows or K23's bits do not hold on this card (or the test errored)"; echo "premise failed rc=$rc" > REFUSAL; finish 25; }
if [ "${P89_PROVE:-0}" = 1 ]; then
  echo "PROVE -- the proving run: install tripwired; premise held; K19 / K16 / K23 contracts compiled on this card" | tee -a summary.txt
  say "PROVE: clone grouped-nf4-gemm @$GNF4_SHA for its contract tests"
  perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4 > logs/clone_gnf4.log 2>&1 && git -C $W/gnf4 checkout -q $GNF4_SHA \
    || { say "PROVE: clone failed"; finish 9; }
  (cd $W/gnf4/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_int4_grouped_smallm_interp.py test_int4_smallm_interp.py -q -p no:cacheprovider) > logs/k19_contract.log 2>&1
  rc=$?; { echo -n "PROVE K19+K16 contract compiled rc=$rc: "; tail -1 logs/k19_contract.log; } | tee -a summary.txt
  [ "$rc" = 0 ] || { say "PROVE: the K19 contract does not hold on this card"; finish 23; }
  (cd $W/gnf4/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_int4_b32.py -k "lean or fused_tile" -q -p no:cacheprovider) > logs/k23_builder.log 2>&1
  rc=$?; { echo -n "PROVE K23 builder compiled rc=$rc: "; tail -1 logs/k23_builder.log; } | tee -a summary.txt
  [ "$rc" = 0 ] || { say "PROVE: K23's builder contract does not hold on this card"; finish 23; }
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
# speed: P88's (P86's) int4 env (RTN experts, uncalibrated int4 attention, glue r1/r2, router epilogue) and command line
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
# speed arm: LEAN=0|1, suffix, census(1|"")
speed(){ local L=$1 SFX=$2 CEN=$3 TAG; TAG=$([ "$L" = 1 ] && echo on || echo off); local AL; AL=$(arm_alarm 1800)
  say "speed $TAG B=16$SFX (census=${CEN:-no} alarm=$AL)"
  env $SPEEDENV E4B_INT4_LEAN_GLUE=$L \
    perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 16 --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --fuse-qkv \
      ${CEN:+--replay-profile-out $W/logs/census_b16_$TAG.txt} --out $W/e4b_b16_$TAG$SFX.json > logs/run_b16_$TAG$SFX.log 2>&1
  local rc=$?
  { echo -n "speed $TAG B=16$SFX rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_b16_$TAG$SFX.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  return $rc; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# the registered order: first draws OFF then ON (censused), second draws ON then OFF
can_run 900 b16_off && { speed 0 "" 1; rec $?; }
can_run 900 b16_on && { speed 1 "" 1; rec $?; }
can_run 900 b16_on_r2 && { speed 1 _r2 ""; rec $?; }
can_run 900 b16_off_r2 && { speed 0 _r2 ""; rec $?; }
say "reduce"; python $W/p89_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
