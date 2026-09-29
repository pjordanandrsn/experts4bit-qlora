#!/bin/bash
# bench/p83/p83_run.sh -- lane P83, BOX side (bench/p83/PREREG-p83.md; e4b#674). A same-box A/B of the licensed build's
# wikitext K8 under two software stacks, both with the fp32 router (E4B_ROUTER_EPI_CAST=0):
#   O = P70's build as P70 ran it: e4b c77aab6 (0.37.4) + grouped-nf4-gemm 5ca1897 (0.33.0), bench/p39's step_decomp and
#       bench/p42's hook (P70's staged bytes), P70's env; then a within-box repeat, P55x's "lic" arm (the expert pack loaded
#       by fingerprint, the attention calibrated live);
#   N = P82's build as P82 ran it: e4b at the launch commit + grouped-nf4-gemm 9407d49 (0.33.7), bench/p81's step_decomp
#       and hook v7, P82's env, both packs dumped; then its repeat, P82's K32 (both packs loaded by fingerprint).
# The preflight, refusals, tripwire style, watchdog and deadline guard are P82's (bench/p82/p82_run.sh). O is installed
# first; N replaces it with --force-reinstall --no-deps (the two cuts declare the same dependencies). The image python
# is used throughout: a venv disables the user site, and the hooks load as usercustomize.
# Started detached by p83_drive.sh with the run's nonce; P83_RUN_NONCE first, then P83_EXIT_CODE.<nonce> + TP_DONE.<nonce>
# on every exit and P83_SUCCESS.<nonce> only when the reducer ran over the four K8 receipts.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P83_MODEL P83_REVISION P83_LICENSED_FP P83_GPU_CLASS P83_MIN_DISK_GB P83_CALIB_NSEQ P83_HESSIAN_BUDGET_GB
# P83_BUILD_PPL_STEPS P83_FIRST_CHUNK_S P83_REHEARSAL.
# P83_PROVE=1 is the PROVING RUN: refusals, BOTH installs at their pins with both tripwires, and an egress probe -- no
# model; exit 0.
set -uo pipefail
W=/root/p83; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p83: $*"; }
NONCE=${P83_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P83_RUN_NONCE.tmp && mv $W/P83_RUN_NONCE.tmp $W/P83_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P83_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P83_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P83_RUN_ID P83_DEADLINE_EPOCH P83_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; done
[ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] || { say "refusing: a pin is not a 40-char sha"; finish 78; }
# ---- the two stacks. O's pins are REGISTERED constants (P70's launch cut and kernel cut); N's come from the launcher.
O_E4B=c77aab6dafb5f08d98a7387d1e986d141b20a2d3; O_GNF4=5ca1897585f9f456f99ea504b2a1be0ea91db496
N_E4B=$E4B_SHA; N_GNF4=$GNF4_SHA
# ---- registered defaults (PREREG-p83 "Instrument"); a rehearsal overrides them and says so
D_MODEL=Qwen/Qwen3-30B-A3B; D_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
D_FP=sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42     # P55x's licensed expert pack
MID=${P83_MODEL:-$D_MODEL}; REV=${P83_REVISION:-$D_REV}; LIC_FP=${P83_LICENSED_FP:-$D_FP}; GPU_CLASS=${P83_GPU_CLASS:-5090}
MIN_DISK_GB=${P83_MIN_DISK_GB:-200}; NSEQ=${P83_CALIB_NSEQ:-128}; HBUDGET=${P83_HESSIAN_BUDGET_GB:-24}
BUILD_STEPS=${P83_BUILD_PPL_STEPS:-2048}; REHEARSAL=${P83_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=$HBUDGET
# every lever is unset; each step sets its own lane's switches. PYTHONPATH (the hook) is set per step, never exported.
unset E4B_SERVE_EXP_INT4 E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB E4B_SERVE_ATTN_INT4 \
      E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_SERVE_ATTN_INT4_ARTIFACT E4B_SERVE_ATTN_INT4_FINGERPRINT E4B_SERVE_ATTN_INT4_DUMP E4B_INT4_ARTIFACT_DIR \
      E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_DUMP_ARTIFACT_DIR E4B_RECOMPILE_LIMIT E4B_ACCUM_RECOMPILE_LIMIT PYTHONPATH
# fp32 routing weights in every process of both stacks: 0.37.4 reads "0" (and unset) as fp32; 0.37.8 reads "0" as fp32
export E4B_ROUTER_EPI_CAST=0
: > summary.txt; echo "$P83_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS router_epi_cast=$E4B_ROUTER_EPI_CAST O=e4b@$O_E4B+gnf4@$O_GNF4 N=e4b@$N_E4B+gnf4@$N_GNF4 model=$MID rev=$REV licensed_fp=$LIC_FP gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB calib_nseq=$NSEQ hessian_budget_gb=$HBUDGET build_ppl_steps=$BUILD_STEPS" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$MID" != "$D_MODEL" ] || [ "$REV" != "$D_REV" ] || [ "$LIC_FP" != "$D_FP" ] || [ "$GPU_CLASS" != 5090 ] \
   || [ "$NSEQ" != 128 ] || [ "$HBUDGET" != 24 ] || [ "$BUILD_STEPS" != 2048 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p83_reduce.py k8_bake.py calib.json o/step_decomp.py o/hook/usercustomize.py n/step_decomp.py n/hook/usercustomize.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p83/staged.sha256"; finish 9; }
python $W/p83_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is fetched: the card class and the disk the working set needs (host-limited: 13)
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the bf16 checkpoint + two NF4 arenas and packs; overlay is not machine disk)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard (P54's): never start a step that cannot finish 10 min before teardown
arm_alarm(){ local left=$(( P83_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
# ---- install one stack: e4b + P37's toolchain pins (image python), gnf4 at its cut. O's install brings the dependencies;
# N's replaces the two packages only (--force-reinstall --no-deps: the cuts declare the same dependencies).
# one pip install with ONE retry after 20 s: a git clone can fail transiently (the A2000 rehearsal's first O install:
# git exit 128; the same pin installed cleanly minutes later). Both attempts' output is kept in the log.
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
install_stack(){ local S=$1 E=$2 G=$3
  say "install stack $S: e4b @$E + gnf4 @$G"
  if [ "$S" = O ]; then
    pipx logs/pip_e4b_$S.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E" \
      "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" || { tail -4 logs/pip_e4b_$S.log; say "PIP FAIL (e4b, $S)"; finish 9; }
  else
    pipx logs/pip_e4b_$S.log 1800 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E" || { tail -4 logs/pip_e4b_$S.log; say "PIP FAIL (e4b, $S)"; finish 9; }
  fi
  pipx logs/pip_gnf4_$S.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$G" || { tail -3 logs/pip_gnf4_$S.log; say "PIP FAIL (gnf4, $S)"; finish 9; }
  # the tripwire: the installed versions ARE the stack's, the router is at fp32, the stack's hook loads as usercustomize
  STACK=$S WANT_E4B=$E WANT_GNF4=$G PYTHONPATH=$W/$(echo $S | tr A-Z a-z)/hook python - <<'PYT' || { say "TRIPWIRE FAIL ($S)"; finish 9; }
import os, sys, json, inspect, importlib.metadata as md
S = os.environ["STACK"]
want = {"O": ("0.37.4", "0.33.0"), "N": (None, "0.33.7")}[S]
import experts4bit_qlora as e, torch, triton, transformers
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"{S}: installed e4b is not the pinned commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"{S}: installed gnf4 is not the pinned commit: {dg}"
if want[0]:
    assert e.__version__ == want[0], f"{S}: e4b {e.__version__} != {want[0]}"
gnf4 = md.version("grouped-nf4-gemm")
assert gnf4 == want[1], f"{S}: gnf4 {gnf4} != {want[1]}"
from experts4bit_qlora.engines import router_epilogue as rte
assert os.environ.get("E4B_ROUTER_EPI_CAST") == "0" and rte.CAST_WEIGHTS[0] is False, \
    f"{S}: the router is not at fp32 (env {os.environ.get('E4B_ROUTER_EPI_CAST')!r}, CAST_WEIGHTS {rte.CAST_WEIGHTS[0]!r})"
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
for k in ("artifact_dir", "expected_fingerprint", "dump_artifact_dir"):
    assert k in inspect.signature(f).parameters, f"{S}: e4b cut lacks the #405 knob {k!r}"
if S == "N":
    from experts4bit_qlora.engines.int4_attn_calib import dump_attn_int4_artifact, enable_serve_attn_int4_from_artifact  # noqa: F401
import site
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
import usercustomize  # noqa: F401
hook = usercustomize.__file__
assert hook.startswith(f"/root/p83/{S.lower()}/hook/"), f"{S}: the hook loaded from {hook}"
open(f"/root/p83/versions_{S}.txt", "w").write(
    f"stack {S}\ne4b {e.__version__} @{os.environ['WANT_E4B']} (from {os.path.dirname(e.__file__)})\ngnf4 {gnf4} @{os.environ['WANT_GNF4']}\n"
    f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n"
    f"cc {torch.cuda.get_device_capability()}\nrouter_epi_cast_weights {rte.CAST_WEIGHTS[0]}\nhook {hook}\n")
print(f"tripwire OK ({S}): e4b", e.__version__, "gnf4", gnf4, "router fp32")
PYT
  cat versions_$S.txt | tee -a summary.txt
  # the stamp itself, exercised under this stack's router module (0.37.4's has no _cast_for), in the proof as in the run
  stamp $W/stamp_$S.json PYTHONPATH=$W/$(echo $S | tr A-Z a-z)/hook
  python -c "import json; d=json.load(open('$W/stamp_$S.json')); assert d['router_epi_cast_weights'] is False, d; print('STAMP $S', d)" | tee -a summary.txt \
    || { say "ROUTER STAMP FAIL ($S)"; finish 9; }; }
# the router stamp beside a K8 receipt: what router_epilogue reads under the step's env, in the stack installed NOW
stamp(){ local out=$1; shift
  env "$@" python - "$out" <<'PYS' || say "router stamp failed for $out"
import json, os, sys, importlib.metadata as md
from experts4bit_qlora.engines import router_epilogue as r
cf = getattr(r, "_cast_for", None)
json.dump({"router_epi_cast_env": os.environ.get("E4B_ROUTER_EPI_CAST"), "router_epi_cast_weights": r.CAST_WEIGHTS[0],
           "router_epi_casts_softmax_topk": (cf("softmax_topk") if cf else bool(r.CAST_WEIGHTS[0])),
           "e4b": md.version("experts4bit-qlora"), "gnf4": md.version("grouped-nf4-gemm"),
           "expected_expert_fingerprint": os.environ.get("E4B_INT4_EXPECTED_FINGERPRINT"),
           "expected_attention_fingerprint": os.environ.get("E4B_SERVE_ATTN_INT4_FINGERPRINT")}, open(sys.argv[1], "w"))
PYS
}
install_stack O "$O_E4B" "$O_GNF4"
if [ "${P83_PROVE:-0}" = 1 ]; then
  install_stack N "$N_E4B" "$N_GNF4"
  echo "PROVE -- the proving run: both stacks installed and tripwired, no model" | tee -a summary.txt
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
# ---- fetch (pinned), once
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
fp_of(){ python -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }
verifies(){ python -c "from experts4bit_qlora.engines.pack_manifest import verify_artifact as v${2:+, $2}; v('$1', expected_model_revision='$REV'${2:+, expected_layout=$2})" > logs/verify_$(basename $1).log 2>&1; }
first_chunk_watchdog(){ local pid=$1 log=$2 budget=${P83_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "$log" 2>/dev/null && { say "build: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
      say "HOST-LIMITED: no calibration chunk in ${budget}s -- killing the build"; echo "HOSTLIMITED build no calibration chunk in ${budget}s" >> summary.txt
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30
    fi
    sleep 20
  done; return 0; }
# ---- one stack's bake, build and repeat. $1 = O|N. Each writes k8_<S>_build.json and k8_<S>_rep.json (+ .router.json).
bake(){ local S=$1 WK=$W/work_$(echo $1 | tr A-Z a-z)
  say "bake NF4 arena ($S)"; mkdir -p $WK
  K8_MODEL="$MID" K8_WORK="$WK" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake_$S.log 2>&1 || { tail -3 logs/bake_$S.log; say "BAKE FAIL ($S)"; finish 12; }
  [ -e "$WK/nf4.arena" ] || { say "BAKE FAIL ($S): no arena"; finish 12; }
  grep -aE "arena|BAKE" logs/bake_$S.log | tail -1 | sed "s/^/$S /" | tee -a summary.txt
  rm -rf $WK/nf4snap 2>/dev/null; }
K8ARGS="--placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps $BUILD_STEPS --b1d-loop eager --no-fuse-qkv --ppl-source wikitext"
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
run_k8(){ local NAME=$1 S=$2 WATCH=$3; shift 3   # the rest: the step's env
  local s=$(echo $S | tr A-Z a-z)
  say "K8 $NAME (stack $S)"
  env PYTHONPATH=$W/$s/hook "$@" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/$s/step_decomp.py --model "$MID" \
      --arena $W/work_$s/nf4.arena --calib $W/calib.json $K8ARGS --out $W/k8_$NAME.json > logs/k8_$NAME.log 2>&1 &
  local pid=$! brc=0 wrc=0
  if [ "$WATCH" = watch ]; then first_chunk_watchdog "$pid" logs/k8_$NAME.log || brc=$?; fi
  wait "$pid" 2>/dev/null || wrc=$?; [ "$brc" = 0 ] && brc=$wrc
  stamp $W/k8_$NAME.router.json PYTHONPATH=$W/$s/hook "$@"
  { echo -n "K8 $NAME rc=$brc "; grep -aE "K8_PPL|REFUSED|Error" logs/k8_$NAME.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt
  [ "$brc" = 30 ] && finish 30
  return 0; }
# ---- stack O: P70's build, verbatim (its env, its harness, the expert pack dumped), then P55x's "lic" repeat
bake O
rm -rf $W/artifact_o
run_k8 O_build O watch E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 \
  E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_CALIB_LAYERS_PER_PASS=10 E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact_o \
  E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 $FOLDS
verifies $W/artifact_o || { say "stack O's build left no expert pack that verifies"; echo "BUILD FAILED (O)" >> summary.txt; finish 20; }
FP_O=$(fp_of $W/artifact_o); [ -n "$FP_O" ] || { say "stack O's pack has no fingerprint"; finish 20; }
echo "PACK O experts $FP_O ($([ "$FP_O" = "$LIC_FP" ] && echo "= the licensed pack" || echo "!= the licensed $LIC_FP"))" | tee -a summary.txt
run_k8 O_rep O nowatch E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 \
  E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_INT4_ARTIFACT_DIR=$W/artifact_o E4B_INT4_EXPECTED_FINGERPRINT=$FP_O \
  E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 $FOLDS
rm -rf $W/work_o 2>/dev/null   # O's arena (~16 GB): nothing later reads it
# ---- stack N: P82's build, verbatim (both packs dumped), then P82's K32 repeat (both packs loaded by fingerprint)
install_stack N "$N_E4B" "$N_GNF4"
bake N
rm -rf $W/artifact_n $W/attn_n
run_k8 N_build N watch \
  E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_CALIB_LAYERS_PER_PASS=10 E4B_CALIB_SOURCE=c4 \
  E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact_n E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_DUMP=$W/attn_n $FOLDS
verifies $W/artifact_n LAYOUT || { say "stack N's build left no expert pack that verifies"; echo "BUILD FAILED (N experts)" >> summary.txt; finish 20; }
verifies $W/attn_n ATTN_LAYOUT || { say "stack N's build left no attention pack that verifies"; echo "BUILD FAILED (N attention)" >> summary.txt; finish 20; }
FP_N=$(fp_of $W/artifact_n); FP_NA=$(fp_of $W/attn_n)
[ -n "$FP_N" ] && [ -n "$FP_NA" ] || { say "a stack-N pack has no fingerprint"; finish 20; }
echo "PACK N experts $FP_N ($([ "$FP_N" = "$LIC_FP" ] && echo "= the licensed pack" || echo "!= the licensed $LIC_FP")); attention $FP_NA" | tee -a summary.txt
run_k8 N_rep N nowatch \
  E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_CALIB_SOURCE=c4 E4B_INT4_ARTIFACT_DIR=$W/artifact_n \
  E4B_INT4_EXPECTED_FINGERPRINT=$FP_N E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_ARTIFACT=$W/attn_n \
  E4B_SERVE_ATTN_INT4_FINGERPRINT=$FP_NA $FOLDS
printf '{"O_expert": "%s", "N_expert": "%s", "N_attention": "%s", "licensed_expert": "%s"}\n' "$FP_O" "$FP_N" "$FP_NA" "$LIC_FP" > packs.json
python $W/p83_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish 0
