#!/bin/bash
# bench/p90/p90_run.sh -- lane P90, BOX side (bench/p90/PREREG-p90.md; e4b#564). Started detached by p90_drive.sh with
# the run's nonce; P90_RUN_NONCE first, then P90_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P90_SUCCESS.<nonce> only when the reducer ran.
#
# Does K21 on gpt-oss-20b's native MXFP4 store (E4B_MXFP4_GROUPED_SMALLM=1, #816 over grouped-nf4-gemm #422/#425) make
# its B=16 decode faster on an RTX 5090 without moving the store's quality? On ONE box:
#   premise  tests/test_k21_row_exact_gpu.py on this card: a token's K21 rows are bit-equal alone (T == 1) and inside a
#            B=16 step, through gpt-oss's epilogue -- so the decode-shaped KL (one token per forward, T == 1, where the
#            opt-in routes K21 too) stands for the batched rows. Run before anything is fetched (rc 25).
#   K0       the KL instrument's own controls on this host (bench/kl_fidelity.py --controls; rc 26 if they fail).
#   speed    B=16 and B=1, the opt-in OFF (0) and ON (1), bo7's store_r12 env (the licensed native MXFP4 store for
#            single rows, NF4 kept for batched rows, folds r1/r2), each drawn twice, the first draw censused.
#            K22's harness command line (graph replay, --no-fuse-qkv).
#   quality  P44's KL-from-reference instrument (bench/p44/kl_serve.py, the gpt-oss reference = dequant of the same
#            shipped MXFP4 bytes, decode-shaped both sides, 200 committed prompts) on arm store_r12, OFF and then ON,
#            the reference scored once and cached.
# The reducer applies the pre-registered rule.
#
# AMENDMENT 1 (2026-10-01): one reading is TWO runs. P90_ARMS=speed (an RTX 5090: the premise, K0, the speed arms) and
# P90_ARMS=quality (an H100 NVL, P44-b's card class: the premise, K0, the KL arms) -- the bf16 dequant reference of
# gpt-oss-20b (~40 GB) does not fit a 32 GB 5090 (p90-5090-1's KL arms died offloading it). Each run writes
# part_<arms>.json; the verdict combines both parts (`p90_reduce.py --speed-dir A --quality-dir B`).
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P90_GPU_CLASS P90_MIN_DISK_GB P90_KL_LIMIT P90_REHEARSAL.
# P90_PROVE=1 is the PROVING RUN: the refusals, the install with its tripwire, the reducer self-test, the premise, the
# K0 controls, grouped-nf4-gemm's K21 + K16 contracts compiled on this card, an egress probe; no model.
set -uo pipefail
W=/root/p90; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p90: $*"; }
NONCE=${P90_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P90_RUN_NONCE.tmp && mv $W/P90_RUN_NONCE.tmp $W/P90_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P90_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P90_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P90_RUN_ID P90_DEADLINE_EPOCH P90_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=4cc831cfa55076f5fe6a23991b88b1c10ca38d87   # grouped-nf4-gemm main (K21 #422 + its masked tail #425 + K24 read #428); a registered constant
MID=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee
ARMS=${P90_ARMS:-}
case "$ARMS" in speed) CLASS_REG=5090;; quality) CLASS_REG="H100 NVL";; *) say "refusing: P90_ARMS must be speed or quality (amendment 1), got '$ARMS'"; finish 78;; esac
GPU_CLASS=${P90_GPU_CLASS:-$CLASS_REG}; MIN_DISK_GB=${P90_MIN_DISK_GB:-150}; KL_LIMIT=${P90_KL_LIMIT:-0}; REHEARSAL=${P90_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export PYTHONPATH=$W/hook E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
# every lever is unset; each arm sets its own switches (E4B_MXFP4_GROUPED_SMALLM explicitly, OFF 0 or ON 1)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_CALIB_SOURCE E4B_CALIB_NSEQ
: > summary.txt; echo "$P90_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS arms=$ARMS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MID rev=$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB kl_limit=$KL_LIMIT" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != "$CLASS_REG" ] || [ "$MIN_DISK_GB" != 150 ] || [ "$KL_LIMIT" != 0 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p90_reduce.py p42_reduce.py test_k21_row_exact_gpu.py \
         serve_stack.py kl_serve.py kl_fidelity.py kl_paths.py kl_prompts.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p90/staged.sha256"; finish 9; }
python $W/p90_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the $ARMS run registers the $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the checkpoint, an NF4 arena, the KL reference cache)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P90_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P90_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P90_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
# one pip install with ONE retry after 20 s: a git clone can fail transiently (P83's A2000 rehearsal: git exit 128)
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
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
import mxfp4_grouped
from experts4bit_qlora.engines import hot_residency as hr
assert hasattr(mxfp4_grouped, "gemm_mxfp4_grouped_smallm") and hr._k21_has_masked_tail(mxfp4_grouped), \
    "gnf4 lacks K21 with its masked K tail (#422 + #425)"
assert hasattr(mxfp4_grouped, "gemv_mxfp4_b32"), "gnf4 lacks the MXFP4 decode GEMV (OFF's single-row route)"
assert hr._K21_PLAN == {"block_n": 32, "kc": 128, "warps": 4, "stages": 3}, f"the route's plan moved: {hr._K21_PLAN}"
assert hr._k21_mode_env() is False, "E4B_MXFP4_GROUPED_SMALLM leaked into the runner's environment"
assert hr._MXFP4_GEMV_ROWS == 16, "the MXFP4 store's GEMV row limit moved: OFF would not be the route bo7 and P44 measured"
import usercustomize  # noqa: F401
assert usercustomize.__file__.startswith("/root/p90/hook/"), usercustomize.__file__
import experts4bit_qlora as e, torch, triton, transformers
from transformers import Mxfp4Config  # noqa: F401  (the gpt-oss dequant reference)
open("/root/p90/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "K21 + masked tail + route present")
PYT
cat versions.txt | tee -a summary.txt
# ---- the premise, on THIS card: K21's rows are row-count invariant through gpt-oss's epilogue. Without it the T == 1 KL
# does not stand for the B=16 rows, so the lane stops before anything is fetched (rc 25).
(cd $W && PYTHONPATH= perl -e 'alarm 900; exec @ARGV' python -m pytest test_k21_row_exact_gpu.py -q -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; { echo -n "PREMISE k21 row-exact rc=$rc: "; tail -1 logs/premise.log; } | tee -a summary.txt
python -c "import json; json.dump({'rc': $rc, 'last': open('$W/logs/premise.log').read().strip().splitlines()[-1:]}, open('$W/premise.json', 'w'))"
[ "$rc" = 0 ] || { say "PREMISE FAILED: K21's rows are not row-count invariant on this card (or the test errored)"; echo "premise failed rc=$rc" > REFUSAL; finish 25; }
# ---- K0: the KL instrument's own controls on THIS host (P44's gate: no KL row without them)
perl -e 'alarm 600; exec @ARGV' python $W/kl_fidelity.py --controls --out $W/k0.json > logs/k0.log 2>&1 \
  && python -c "import json; r=json.load(open('$W/k0.json')); assert r['all_passed']" 2>/dev/null \
  || { tail -5 logs/k0.log; say "K0 CONTROLS FAILED -- no KL row is produced (the instrument's rule)"; echo "K0 FAILED" >> summary.txt; finish 26; }
echo "K0 all_passed" | tee -a summary.txt
if [ "${P90_PROVE:-0}" = 1 ]; then
  echo "PROVE -- the proving run: install tripwired; premise held; K0 passed; K21 + K16 contracts compiled on this card" | tee -a summary.txt
  say "PROVE: clone grouped-nf4-gemm @$GNF4_SHA for its contract tests"
  perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4 > logs/clone_gnf4.log 2>&1 && git -C $W/gnf4 checkout -q $GNF4_SHA \
    || { say "PROVE: clone failed"; finish 9; }
  (cd $W/gnf4/kernel && PYTHONPATH= TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_mxfp4_grouped_smallm_interp.py test_int4_smallm_interp.py -q -p no:cacheprovider) > logs/k21_contract.log 2>&1
  rc=$?; { echo -n "PROVE K21+K16 contract compiled rc=$rc: "; tail -1 logs/k21_contract.log; } | tee -a summary.txt
  [ "$rc" = 0 ] || { say "PROVE: the K21 contract does not hold on this card"; finish 23; }
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
# ---- fetch (pinned; the shipped MXFP4 bytes, not the original/ or metal/ trees), bake (P39's k8_bake.py)
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', ignore_patterns=['original/*', 'metal/*'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
say "bake NF4 arena"; mkdir -p $W/work_gptoss
K8_MODEL="$MID" K8_WORK="$W/work_gptoss" perl -e "alarm $(arm_alarm 2400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work_gptoss/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL: no arena"; finish 12; }
# speed: bo7's store_r12 (bench/hybrid-g9/throughput-20260904/bo7/logs/bo7_run.sh), the env K22/K24 censused
STOREENV="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=0"
# speed arm: K21=0|1, B, suffix, census(1|"")
speed(){ local K=$1 B=$2 SFX=$3 CEN=$4 TAG; TAG=$([ "$K" = 1 ] && echo on || echo off); local AL; AL=$(arm_alarm 1800)
  say "speed $TAG B=$B$SFX (census=${CEN:-no} alarm=$AL)"
  env $STOREENV E4B_MXFP4_GROUPED_SMALLM=$K \
    perl -e "alarm $AL; exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed --no-fuse-qkv \
      ${CEN:+--replay-profile-out $W/logs/census_b${B}_$TAG.txt} --out $W/e4b_b${B}_$TAG$SFX.json > logs/run_b${B}_$TAG$SFX.log 2>&1
  local rc=$?
  { echo -n "speed $TAG B=$B$SFX rc=$rc "; grep -aE "B1D_TIMED|BV3_" logs/run_b${B}_$TAG$SFX.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  return $rc; }
# quality arm: K21=0|1 -> gptoss_kl_<tag>.json (P44's kl_serve, arm store_r12). OFF runs control (i) (`--scorer auto`:
# decode when the reference agrees with itself, as P44 read for gpt-oss), scores the reference once and caches it; ON
# reuses that decode cache (`--scorer decode`, no reference reload). The reducer VOIDs unless both rows are decode.
# --limit is a REHEARSAL knob only.
kl(){ local K=$1 TAG; TAG=$([ "$K" = 1 ] && echo on || echo off); local AL; AL=$(arm_alarm 3600)
  say "kl $TAG (store_r12, decode-shaped; alarm=$AL)"
  E4B_MXFP4_GROUPED_SMALLM=$K E4B_MODEL_ID=$MID perl -e "alarm $AL; exec @ARGV" python -u $W/kl_serve.py --family gptoss --arms store_r12 \
      --arena "$QA" --calib $W/calib.json --k0-receipt $W/k0.json --ref-cache $W/refcache_gptoss --scorer $([ "$K" = 1 ] && echo decode || echo auto) \
      --controls $([ "$K" = 1 ] && echo 0 || echo 1) --limit $KL_LIMIT --out $W/gptoss_kl_$TAG.json > logs/kl_$TAG.log 2>&1
  local rc=$?
  { echo -n "kl $TAG rc=$rc "; grep -aE "^== " logs/kl_$TAG.log | tail -1 | cut -c1-200; echo; } >> summary.txt
  return $rc; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# the registered order. speed (5090): first draws interleaved OFF/ON (censused), then second draws ON/OFF. quality (H100
# NVL): KL OFF (scores and caches the reference), then KL ON.
if [ "$ARMS" = speed ]; then
  for B in 16 1; do
    can_run 900 b${B}_off && { speed 0 $B "" 1; rec $?; }
    can_run 900 b${B}_on && { speed 1 $B "" 1; rec $?; }
  done
  for B in 16 1; do
    can_run 900 b${B}_on_r2 && { speed 1 $B _r2 ""; rec $?; }
    can_run 900 b${B}_off_r2 && { speed 0 $B _r2 ""; rec $?; }
  done
else
  can_run 3000 kl_off && { kl 0; rec $?; }
  can_run 1800 kl_on && { kl 1; rec $?; }
fi
say "reduce ($ARMS part)"; python $W/p90_reduce.py --part $ARMS --dir $W --out $W/part_$ARMS.json 2>&1 | tee -a summary.txt
[ -s $W/part_$ARMS.json ] || { say "REDUCER wrote no part report"; finish 22; }
finish "$rc_any"
