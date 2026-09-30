#!/bin/bash
# bench/p85/p85_run.sh -- lane P85, BOX side (bench/p85/PREREG-p85.md; e4b#674). Is grouped-nf4-gemm#413 (0.33.7: the
# fused fp8 KV append's quotient IEEE-rounded, div_rn) the whole KERNEL step lane P84 found? P84: on one box,
# grouped-nf4-gemm 0.33.0 -> 0.33.7 moved the recipe's fp32-router wikitext K8 from O's 1.8511420498367808 to N's
# 1.8506507749113845; the e4b release and the harness moved it by zero. Everything here runs on ONE AMD-host box, through
# P70's harness (bench/p39/step_decomp.py + bench/p42 hook v6, o/) and P70's env, E4B_ROUTER_EPI_CAST=0, e4b c77aab6:
#   O_build (control) = P70's build exactly: stack O = e4b c77aab6 (0.37.4) + gnf4 5ca1897 (0.33.0); the expert pack
#       dumped. Must equal O's known mean NLL bit for bit, or the lane is VOID and stops. O_rep = P55x's lic arm (the
#       expert pack loaded by fingerprint, the attention calibrated live).
#   F_1, F_2 = the lic arm on stack O with E4B_FUSED_KV_APPEND=0: every KV append through quantize_kv_fp8, whose bytes
#       0.33.7's fused kernel writes exactly (lane B771). Predicted: N's known float.
#   S_1, S_2 = the lic arm on stack S = the same e4b + gnf4 f879761 (0.33.6: every release of the cut but #413's), the
#       fused append on (its default). Predicted: O's known float.
# Preflight, refusals (card class, AMD host -- P84's amendment 1 -- and disk), tripwires, stamps, the first-chunk watchdog
# and the deadline guard are P84's (bench/p84/p84_run.sh). Stack O is installed first with the dependencies; S replaces
# grouped-nf4-gemm alone (--force-reinstall --no-deps). The launch commit (E4B_SHA) stages this kit and is recorded; the
# box never installs it. Started detached by p85_drive.sh with the run's nonce; P85_RUN_NONCE first, then
# P85_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P85_SUCCESS.<nonce> only when the reducer ran.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P85_MODEL P85_REVISION P85_LICENSED_FP P85_GPU_CLASS P85_CPU_VENDOR P85_MIN_DISK_GB P85_CALIB_NSEQ P85_HESSIAN_BUDGET_GB
# P85_BUILD_PPL_STEPS P85_FIRST_CHUNK_S P85_REHEARSAL.
# P85_PROVE=1 is the PROVING RUN: refusals, BOTH installs at their pins with their tripwires (P70's harness importing
# under each) and stamps (the fused append resolving on, and off under the knob), and an egress probe -- no model; exit 0.
set -uo pipefail
W=/root/p85; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p85: $*"; }
NONCE=${P85_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P85_RUN_NONCE.tmp && mv $W/P85_RUN_NONCE.tmp $W/P85_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P85_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P85_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P85_RUN_ID P85_DEADLINE_EPOCH P85_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
# ---- the two stacks: REGISTERED constants. O = P70's build (e4b 0.37.4 + gnf4 v0.33.0); S = the same e4b + gnf4 v0.33.6.
O_E4B=c77aab6dafb5f08d98a7387d1e986d141b20a2d3; O_GNF4=5ca1897585f9f456f99ea504b2a1be0ea91db496
S_E4B=$O_E4B; S_GNF4=f879761501fedcac027de81f4083fab924194b8d
# ---- registered defaults (PREREG-p85 "Instrument"); a rehearsal overrides them and says so
D_MODEL=Qwen/Qwen3-30B-A3B; D_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
D_FP=sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42     # P55x's licensed expert pack
MID=${P85_MODEL:-$D_MODEL}; REV=${P85_REVISION:-$D_REV}; LIC_FP=${P85_LICENSED_FP:-$D_FP}; GPU_CLASS=${P85_GPU_CLASS:-5090}
# P84's amendment 1: the known floats were read on AMD hosts only (an Intel i9-14900K calibrated another attention pack)
CPU_VENDOR=${P85_CPU_VENDOR:-AuthenticAMD}
MIN_DISK_GB=${P85_MIN_DISK_GB:-200}; NSEQ=${P85_CALIB_NSEQ:-128}; HBUDGET=${P85_HESSIAN_BUDGET_GB:-24}
BUILD_STEPS=${P85_BUILD_PPL_STEPS:-2048}; REHEARSAL=${P85_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=$HBUDGET
# every lever is unset; each step sets its own switches. PYTHONPATH (the hook) is set per step, never exported.
# E4B_FUSED_KV_APPEND is unset here so every reading but F's takes the fused append by its default, as P70's did.
unset E4B_SERVE_EXP_INT4 E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB E4B_SERVE_ATTN_INT4 \
      E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST E4B_FUSED_KV_APPEND \
      E4B_SERVE_ATTN_INT4_ARTIFACT E4B_SERVE_ATTN_INT4_FINGERPRINT E4B_SERVE_ATTN_INT4_DUMP E4B_INT4_ARTIFACT_DIR \
      E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_DUMP_ARTIFACT_DIR E4B_RECOMPILE_LIMIT E4B_ACCUM_RECOMPILE_LIMIT PYTHONPATH
# fp32 routing weights in every process: 0.37.4 reads "0" (and unset) as fp32
export E4B_ROUTER_EPI_CAST=0
: > summary.txt; echo "$P85_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS router_epi_cast=$E4B_ROUTER_EPI_CAST launch=e4b@$E4B_SHA O=e4b@$O_E4B+gnf4@$O_GNF4 S=e4b@$S_E4B+gnf4@$S_GNF4 model=$MID rev=$REV licensed_fp=$LIC_FP gpu_class=$GPU_CLASS cpu_vendor=$CPU_VENDOR min_disk_gb=$MIN_DISK_GB calib_nseq=$NSEQ hessian_budget_gb=$HBUDGET build_ppl_steps=$BUILD_STEPS" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$MID" != "$D_MODEL" ] || [ "$REV" != "$D_REV" ] || [ "$LIC_FP" != "$D_FP" ] || [ "$GPU_CLASS" != 5090 ] || [ "$CPU_VENDOR" != AuthenticAMD ] \
   || [ "$NSEQ" != 128 ] || [ "$HBUDGET" != 24 ] || [ "$BUILD_STEPS" != 2048 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p85_reduce.py k8_bake.py calib.json o/step_decomp.py o/hook/usercustomize.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p85/staged.sha256"; finish 9; }
python $W/p85_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is fetched: the card class, the host CPU vendor, the disk (host-limited: 13)
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
HOST_VENDOR=$(lscpu | awk -F: '/^Vendor ID/{gsub(/ /, "", $2); print $2; exit}')
echo "cpu_vendor $HOST_VENDOR" | tee -a forensics.txt
[ "$HOST_VENDOR" = "$CPU_VENDOR" ] || { say "REFUSED: host CPU vendor is '${HOST_VENDOR:-unknown}', the lane registers $CPU_VENDOR (P84 amendment 1)"; echo "refused: cpu vendor ${HOST_VENDOR:-unknown}" > REFUSAL; finish 16; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the bf16 checkpoint, an NF4 arena and the expert pack; overlay is not machine disk)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard (P54's): never start a step that cannot finish 10 min before teardown
arm_alarm(){ local left=$(( P85_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
# one pip install with ONE retry after 20 s: a git clone can fail transiently (P83's A2000 rehearsal: git exit 128)
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
# ---- install one stack. O brings e4b c77aab6 with P37's toolchain pins (image python) and the dependencies, then gnf4 at
# its cut; S replaces gnf4 alone (--force-reinstall --no-deps: the two cuts declare the same dependencies).
install_stack(){ local S=$1 E=$2 G=$3
  say "install stack $S: e4b @$E + gnf4 @$G"
  if [ "$S" = O ]; then
    pipx logs/pip_e4b_$S.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E" \
      "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" || { tail -4 logs/pip_e4b_$S.log; say "PIP FAIL (e4b, $S)"; finish 9; }
  fi
  pipx logs/pip_gnf4_$S.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$G" || { tail -3 logs/pip_gnf4_$S.log; say "PIP FAIL (gnf4, $S)"; finish 9; }
  # the tripwire: the installed versions ARE the stack's; the router is at fp32; the installed fp8_kv is the pre-#413
  # kernel; the fused append resolves ON by default and OFF under the knob; P70's hook and step_decomp load
  STACK=$S WANT_E4B=$E WANT_GNF4=$G PYTHONPATH=$W/o/hook python - <<'PYT' || { say "TRIPWIRE FAIL ($S)"; finish 9; }
import os, sys, json, inspect, importlib.metadata as md
S = os.environ["STACK"]
want = {"O": ("0.37.4", "0.33.0"), "S": ("0.37.4", "0.33.6")}[S]
import experts4bit_qlora as e, torch, triton, transformers
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"{S}: installed e4b is not the pinned commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"{S}: installed gnf4 is not the pinned commit: {dg}"
assert e.__version__ == want[0], f"{S}: e4b {e.__version__} != {want[0]}"
gnf4 = md.version("grouped-nf4-gemm")
assert gnf4 == want[1], f"{S}: gnf4 {gnf4} != {want[1]}"
from experts4bit_qlora.engines import router_epilogue as rte
assert os.environ.get("E4B_ROUTER_EPI_CAST") == "0" and rte.CAST_WEIGHTS[0] is False, \
    f"{S}: the router is not at fp32 (env {os.environ.get('E4B_ROUTER_EPI_CAST')!r}, CAST_WEIGHTS {rte.CAST_WEIGHTS[0]!r})"
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
for k in ("artifact_dir", "expected_fingerprint", "dump_artifact_dir"):
    assert k in inspect.signature(f).parameters, f"{S}: e4b cut lacks the #405 knob {k!r}"
import fp8_kv
from fp8_kv import fp8_kv_append_t1  # noqa: F401  (the fused append exists in this cut)
assert not hasattr(fp8_kv, "_quantize_kv_fp32"), f"{S}: fp8_kv carries #413 (_quantize_kv_fp32) -- not a pre-0.33.7 kernel"
from experts4bit_qlora.engines.fp8_paged_kv import _resolve_fused_append as res
assert res(None, "cuda", lambda: True) is True and res("0", "cuda", lambda: True) is False, f"{S}: the append resolver moved"
import site
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
import usercustomize  # noqa: F401
hook = usercustomize.__file__
assert hook.startswith("/root/p85/o/hook/"), f"{S}: the hook loaded from {hook}"
sys.path.insert(0, "/root/p85/o")
import step_decomp  # noqa: F401  (P70's harness copy imports under this stack)
assert step_decomp.__file__.startswith("/root/p85/o/"), step_decomp.__file__
open(f"/root/p85/versions_{S}.txt", "w").write(
    f"stack {S}\ne4b {e.__version__} @{os.environ['WANT_E4B']} (from {os.path.dirname(e.__file__)})\ngnf4 {gnf4} @{os.environ['WANT_GNF4']}\n"
    f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n"
    f"cc {torch.cuda.get_device_capability()}\nrouter_epi_cast_weights {rte.CAST_WEIGHTS[0]}\nfp8_kv_has_413 False\nhook {hook}\n")
print(f"tripwire OK ({S}): e4b", e.__version__, "gnf4", gnf4, "router fp32, pre-#413 fp8_kv, append resolver on/off")
PYT
  cat versions_$S.txt | tee -a summary.txt
  # the stamp itself, exercised under this stack in the proof as in the run: once by default, once under the knob
  stamp $W/stamp_$S.json PYTHONPATH=$W/o/hook
  stamp $W/stamp_${S}_append_off.json PYTHONPATH=$W/o/hook E4B_FUSED_KV_APPEND=0
  python -c "
import json
d = json.load(open('$W/stamp_$S.json')); x = json.load(open('$W/stamp_${S}_append_off.json'))
assert d['router_epi_cast_weights'] is False and x['router_epi_cast_weights'] is False, (d, x)
assert d['fused_kv_append_resolved'] is True and x['fused_kv_append_resolved'] is False, (d, x)
assert d['fp8_kv_has_413'] is False, d
print('STAMP $S', d); print('STAMP ${S} append off', x)" | tee -a summary.txt || { say "STAMP FAIL ($S)"; finish 9; }; }
# the stamp beside a K8 receipt: what the router module and the fused-append resolver read under the step's env, in the
# stack installed NOW (0.37.4's router module has no _cast_for)
stamp(){ local out=$1; shift
  env "$@" python - "$out" <<'PYS' || say "stamp failed for $out"
import json, os, sys, importlib.metadata as md
import torch
from experts4bit_qlora.engines import router_epilogue as r
from experts4bit_qlora.engines.fp8_paged_kv import _resolve_fused_append
import fp8_kv
def present():
    try:
        from fp8_kv import fp8_kv_append_t1  # noqa: F401
        return True
    except ImportError:
        return False
cf = getattr(r, "_cast_for", None)
env = os.environ.get("E4B_FUSED_KV_APPEND")
json.dump({"router_epi_cast_env": os.environ.get("E4B_ROUTER_EPI_CAST"), "router_epi_cast_weights": r.CAST_WEIGHTS[0],
           "router_epi_casts_softmax_topk": (cf("softmax_topk") if cf else bool(r.CAST_WEIGHTS[0])),
           "fused_kv_append_env": env,
           "fused_kv_append_resolved": _resolve_fused_append(env, "cuda" if torch.cuda.is_available() else "cpu", present),
           "fp8_kv_has_413": hasattr(fp8_kv, "_quantize_kv_fp32"),
           "e4b": md.version("experts4bit-qlora"), "gnf4": md.version("grouped-nf4-gemm"),
           "expected_expert_fingerprint": os.environ.get("E4B_INT4_EXPECTED_FINGERPRINT")}, open(sys.argv[1], "w"))
PYS
}
install_stack O "$O_E4B" "$O_GNF4"
if [ "${P85_PROVE:-0}" = 1 ]; then
  install_stack S "$S_E4B" "$S_GNF4"
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
verifies(){ python -c "from experts4bit_qlora.engines.pack_manifest import verify_artifact as v; v('$1', expected_model_revision='$REV')" > logs/verify_$(basename $1).log 2>&1; }
first_chunk_watchdog(){ local pid=$1 log=$2 budget=${P85_FIRST_CHUNK_S:-900} t0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "$log" 2>/dev/null && { say "build: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
      say "HOST-LIMITED: no calibration chunk in ${budget}s -- killing the build"; echo "HOSTLIMITED build no calibration chunk in ${budget}s" >> summary.txt
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30
    fi
    sleep 20
  done; return 0; }
# ---- the steps. $1 = the reading, $2 = the harness dir (o = P70's, the only one here), $3 = the arena dir,
# $4 = watch|nowatch, the rest = the step's env. Each writes k8_<reading>.json (+ .router.json).
bake(){ local WK=$W/$1
  say "bake NF4 arena ($1)"; mkdir -p $WK
  K8_MODEL="$MID" K8_WORK="$WK" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake_$1.log 2>&1 || { tail -3 logs/bake_$1.log; say "BAKE FAIL ($1)"; finish 12; }
  [ -e "$WK/nf4.arena" ] || { say "BAKE FAIL ($1): no arena"; finish 12; }
  grep -aE "arena|BAKE" logs/bake_$1.log | tail -1 | sed "s/^/$1 /" | tee -a summary.txt
  rm -rf $WK/nf4snap 2>/dev/null; }
K8ARGS="--placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps $BUILD_STEPS --b1d-loop eager --no-fuse-qkv --ppl-source wikitext"
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
run_k8(){ local NAME=$1 H=$2 WK=$3 WATCH=$4; shift 4
  say "K8 $NAME (harness $H, arena $WK)"
  env PYTHONPATH=$W/$H/hook "$@" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/$H/step_decomp.py --model "$MID" \
      --arena $W/$WK/nf4.arena --calib $W/calib.json $K8ARGS --out $W/k8_$NAME.json > logs/k8_$NAME.log 2>&1 &
  local pid=$! brc=0 wrc=0
  if [ "$WATCH" = watch ]; then first_chunk_watchdog "$pid" logs/k8_$NAME.log || brc=$?; fi
  wait "$pid" 2>/dev/null || wrc=$?; [ "$brc" = 0 ] && brc=$wrc
  stamp $W/k8_$NAME.router.json PYTHONPATH=$W/$H/hook "$@"
  { echo -n "K8 $NAME rc=$brc "; grep -aE "K8_PPL|REFUSED|Error" logs/k8_$NAME.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt
  [ "$brc" = 30 ] && finish 30
  return 0; }
# P55x's lic arm (P70's repeat): the O build's expert pack loaded by fingerprint, the attention calibrated live; the rest
# of the arguments are this reading's own env (F's knob), appended last so it is the one in force
lic(){ local NAME=$1 WK=$2; shift 2
  run_k8 $NAME o $WK nowatch E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 \
    E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_INT4_ARTIFACT_DIR=$W/artifact_o E4B_INT4_EXPECTED_FINGERPRINT=$FP_O \
    E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 $FOLDS "$@"; }
# ---- stack O: the control O_build (P70's build env, the expert pack dumped) first
bake work_o
rm -rf $W/artifact_o
run_k8 O_build o work_o watch E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 \
  E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_CALIB_LAYERS_PER_PASS=10 E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact_o \
  E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 $FOLDS
verifies $W/artifact_o || { say "O_build left no expert pack that verifies"; echo "BUILD FAILED (O experts)" >> summary.txt; finish 20; }
FP_O=$(fp_of $W/artifact_o); [ -n "$FP_O" ] || { say "O_build's pack has no fingerprint"; finish 20; }
mkdir -p $W/manifests && cp $W/artifact_o/manifest.json $W/manifests/O_build_experts.json   # the pack is deleted at the end
echo "PACK O_build experts $FP_O ($([ "$FP_O" = "$LIC_FP" ] && echo "= the licensed pack" || echo "!= the licensed $LIC_FP"))" | tee -a summary.txt
# the control gates the rest: if this box does not give O's known float, no comparison with a known float holds
if ! python $W/p85_reduce.py --control-ok $W/k8_O_build.json | tee -a summary.txt; then
  say "CONTROL: this box does not reproduce O's known K8 -- VOID; stopping before the remaining readings"
  python $W/p85_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
  [ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
  finish 0
fi
lic O_rep work_o
# F: the same stack and pack, every KV append through quantize_kv_fp8 (the fused append resolved off)
lic F_1 work_o E4B_FUSED_KV_APPEND=0
lic F_2 work_o E4B_FUSED_KV_APPEND=0
# ---- stack S: gnf4 0.33.6 over the same e4b, its own bake, the same expert pack, the fused append on
install_stack S "$S_E4B" "$S_GNF4"
rm -rf $W/work_o
bake work_s
lic S_1 work_s
lic S_2 work_s
printf '{"O_expert": "%s", "licensed_expert": "%s"}\n' "$FP_O" "$LIC_FP" > packs.json
rm -rf $W/artifact_o $W/work_s   # the pack (~16 GB) and the arena: nothing later reads them; the manifest travels
python $W/p85_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish 0
