#!/bin/bash
# Lane P223 (bench/p223/PREREG-p223.md): the price of a schedule-invariant decode for Mixtral-8x7B on serve_paged, on ONE
# RTX 5090 at >= 575 W. P149's box Q+S runner (bench/p149/p149_run.sh) by substitution: the arms (D0, I0, D1, I1), no Q
# and its early gates, no token-pinned processes, no SANE phases, two ABBA blocks of fresh processes, no dry mode.
#   floors   CUDA 18, the card class and cc 12.0 15, the disk 13, host RAM 16, power.limit AND enforced.power.limit
#            >= 575 W 17 (unreadable 15), all before the install; then the guard fit over every term below at its
#            allowance (rc 19), also before the install;
#   premise  on THIS card: the grouped-nf4-gemm K37 suite compiled (28 passed) and the e4b int8 kernel path compiled (4
#            passed), rc 25 (D1 and I1 serve the int8 projections);
#   order    install + tripwire; reducer and box self-tests; premise; fetch; NF4 arena bake; prompts; then D0a, I0a, I1a,
#            D1a, D1b, I1b, I0b, D0b, each a fresh process; the first process of each arm is read against the rule's
#            VOID checks before the next (rc 45: the box stops, the reduce still writes the VOID); then the reduce. Every
#            phase stamps its UTC start and end in phases.jsonl.
# Refusals: as above. Only 13 and 18 name the machine for the next launcher draw. This script never creates compute.
set -uo pipefail
W=/root/p223; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p223: $*"; }
NONCE=${P223_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P223_RUN_NONCE.tmp && mv $W/P223_RUN_NONCE.tmp $W/P223_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P223_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P223_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P223_RUN_ID P223_DEADLINE_EPOCH P223_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7 # v0.49.0 tag commit: e4b's CI pin, the release that ships K37 (dense_i8)
GNF4_VERSION=0.49.0
SHORT=32; LONG=160; REPS=5
# the registration's fit table: the allowances every time-left check uses (s); boot (300) is spent before this runs
INSTALL=450; NEED_FETCH=3000; NEED_BAKE=1500; NEED_ARM=480; NEED_REDUCE=120
RESERVE=600
GPU_CLASS=5090; MIN_RAM_GB=60; MIN_DISK_GB=200; MIN_POWER_W=575
case "$P223_RUN_ID" in
  p223-5090-1|p223-5090-2|p223-5090-3) ;;
  *) say "unregistered run ID: $P223_RUN_ID"; finish 78;;
esac
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; REV=eba92302a2861cdc0098cc54bc9f17cb2c47eb61   # the P141 lock
[ -z "${P223_GPU_CLASS:-}${P223_MIN_DISK_GB:-}${P223_MIN_RAM_GB:-}${P223_REHEARSAL:-}${P223_SHORT:-}${P223_LONG:-}${P223_REPS:-}${P223_MIN_POWER_W:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject (the arms set their knobs alone)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_ATTN_SPLIT_CAP \
      E4B_SERVE_ATTN_INT8 E4B_DENSE_GEMV_BW E4B_INT8_HEAD_BW
: > summary.txt; : > phases.jsonl; echo "$P223_INSTANCE_ID" > INSTANCE_ID; echo "$P223_RUN_ID" > RUN_ID
echo "KNOBS run=$P223_RUN_ID e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W short=$SHORT long=$LONG reps=$REPS" | tee -a summary.txt
stamp(){ printf '{"phase": "%s", "event": "%s", "t": %s}\n' "$1" "$2" "$(date +%s)" >> phases.jsonl; }
# ---- staged pieces, byte-for-byte
for f in p223_box.py p223_reduce.py p109_box.py k8_bake.py calib.json test_dense_i8.py test_int8_attn_interp.py staged-p223.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p223.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p223/staged-p223.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the card class, the disk, the host RAM, the power floor
CUDA_PROBE=$(python - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
try:
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
except Exception:
    ok = False
print("ok" if ok else f"no-cuda torch {torch.__version__}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE}) -- registered host floor"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm --format=csv,noheader | sed "s/^/power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; finish 15; }
SMS=$(python -c "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)" 2>/dev/null)
[ "$SMS" = 170 ] || { say "REFUSED: $SMS SMs, the lane registers the 170-SM RTX 5090"; echo "refused: sms $SMS" > REFUSAL; finish 18; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: 5090 VRAM ${VRAM_MIB:-?} MiB below 30000 MiB floor"; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the 93.4 GB checkpoint, its NF4 snapshot and arena, the venv)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
EPL=$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
[ -n "$PL" ] && [ -n "$EPL" ] || { say "REFUSED: power.limit or enforced.power.limit unreadable"; echo "refused: power limits unreadable" > REFUSAL; finish 15; }
echo "POWER_LIMIT_W $PL ENFORCED_W $EPL (floor $MIN_POWER_W)" | tee -a summary.txt
[ "$PL" -ge "$MIN_POWER_W" ] && [ "$EPL" -ge "$MIN_POWER_W" ] || { say "REFUSED: power.limit ${PL} W / enforced ${EPL} W < ${MIN_POWER_W} W (host-limited power cap)"; echo "refused: power.limit ${PL} W enforced ${EPL} W" > REFUSAL; finish 17; }
# the registration's fit, before the install, in the cross-lane form (bench/common/README.md, "Guard fit"): everything
# from here to the verdict at its allowance, plus the reserve, with 25 % headroom (rounded up: 9,510 x 1.25 = 11,888 s)
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL + NEED_FETCH + NEED_BAKE + 8 * NEED_ARM + NEED_REDUCE + RESERVE)); FIT_NEED=$(((FIT * FIT_HEADROOM_PCT + 99) / 100)); LEFT=$((P223_DEADLINE_EPOCH - $(date +%s)))
echo "GUARD_FIT ${FIT}s x ${FIT_HEADROOM_PCT}% = ${FIT_NEED}s of ${LEFT}s left" | tee -a summary.txt
# rc 19: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: guard fit ${LEFT}s < ${FIT_NEED}s" > REFUSAL; finish 19; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + RESERVE)) -le "$P223_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s + ${RESERVE}s reserve = $((need + RESERVE))s, only $((P223_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P223_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
TORCH_PIN=2.8.0; echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
stamp install start
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_GNF4_VERSION=$GNF4_VERSION python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import inspect, os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == os.environ["WANT_GNF4_VERSION"], md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import triton
assert triton.__version__ == "3.4.0", triton.__version__
# the int8 lever: K37's entry point, e4b's engine, the knob in the snapshot, the census for mixtral
from dense_i8 import linear_i8, quantize_rows_int8  # noqa: F401
from experts4bit_qlora.engines import int8_attn
assert int8_attn.CENSUS == {"mixtral": ("q_proj", "k_proj", "v_proj", "o_proj")}, int8_attn.CENSUS
from experts4bit_qlora import serve_paged as sp
assert "E4B_SERVE_ATTN_INT8" in sp.LEVER_ENV and hasattr(sp, "_attn_int8_census")
assert sp._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
m, s = sp.resolve_fusion_modes({k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}, "mixtral")
assert set(m.values()) == {"auto"} and set(s.values()) == {"default-allowlisted"}, (m, s)
# the invariance knobs: the cap reaches the kernel, mixtral's unset cap is the library's plan, the buckets as registered
assert sp._attn_split_cap_env("1") == "1" and sp.resolve_attn_split_cap(sp.FUSION_UNSET, "mixtral") == (None, "default-off")
import fp8_paged_attn
assert "split_cap" in inspect.signature(fp8_paged_attn.fp8_paged_decode_attention).parameters
assert "split_cap" in inspect.signature(fp8_paged_attn.decode_split_plan).parameters
from experts4bit_qlora.serve_recipe import default_buckets, usable_buckets
assert default_buckets(16) == (1, 2, 4, 8, 16) and usable_buckets(16, (2, 4, 8, 16)) == (2, 4, 8, 16)
import torch
import experts4bit_qlora as e
open("/root/p223/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p223_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p223_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: K37's suite and e4b's kernel path, compiled
(cd $W && PYTHONPATH='' TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_dense_i8.py -q -rs -p no:cacheprovider) > logs/premise_k37.log 2>&1
rc1=$?; L1=$(tail -1 logs/premise_k37.log)
(cd $W && PYTHONPATH='' TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_int8_attn_interp.py -q -rs -p no:cacheprovider) > logs/premise_engine.log 2>&1
rc2=$?; L2=$(tail -1 logs/premise_engine.log)
{ echo "premise K37 rc=$rc1: $L1"; echo "premise engine rc=$rc2: $L2"; } | tee -a summary.txt
if [ "$rc1" = 0 ] && echo "$L1" | grep -qE '^28 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$' \
   && [ "$rc2" = 0 ] && echo "$L2" | grep -qE '^4 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed" > REFUSAL; finish 25
fi
stamp install end
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as P145 bakes it) and the prompts
can_run $NEED_FETCH fetch || finish 40
stamp fetch start; say "fetch $MODEL @ $REV"
PATTERNS="ignore_patterns=['original/*','metal/*','consolidated*']"
perl -e "alarm $(step_alarm $NEED_FETCH); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', $PATTERNS, max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt; stamp fetch end
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
stamp bake start; say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm $NEED_BAKE); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt; stamp bake end
perl -e "alarm 1200; exec @ARGV" python $W/p223_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# the D arms run today's buckets (auto = 1,2,4,8,16 at 16 slots), so E4B_PAGED_BUCKETS is an I-arm knob, never the engine's
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768"
arm_knobs(){   # ARM: the registration's knobs, each in the process environment
  case "$1" in
    D0) echo "";;
    I0) echo "E4B_ATTN_SPLIT_CAP=1 E4B_PAGED_BUCKETS=2,4,8,16";;
    D1) echo "E4B_SERVE_ATTN_INT8=1";;
    I1) echo "E4B_SERVE_ATTN_INT8=1 E4B_ATTN_SPLIT_CAP=1 E4B_PAGED_BUCKETS=2,4,8,16";;
  esac
}
run_arm(){   # TAG: one fresh speed process, five timed reps at W16 and W1
  local TAG=$1 ARM=${1:0:2} LV AL
  LV=$(arm_knobs "$ARM")
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm $((NEED_ARM + 600))); say "arm $TAG ${LV:-(the default server)} (alarm=$AL)"; stamp "$TAG" start
  # shellcheck disable=SC2086  # ENGINE_ENV and LV are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $LV P223_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p223_box.py --mode speed --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  local rc=$?; stamp "$TAG" end
  { echo -n "arm $TAG rc=$rc "; grep -a "^P223_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-400 | tr '\n' ' '; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
}
reduce_all(){
  python $W/p223_reduce.py --dir $W --out $W/verdict_p223.json --e4b-sha "$E4B_SHA" 2>&1 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p223.json" ]
}
gate_check(){   # TAG: the first process of an arm, read against the rule's VOID checks before the next process
  python $W/p223_reduce.py --gate $W/arm_$1.json --e4b-sha "$E4B_SHA" 2>&1 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] && return 0
  say "EARLY VOID ($1): its record fails the rule's checks -- the box stops; the reduce writes the VOID"
  echo "early void: $1" > REFUSAL; stamp reduce start; reduce_all; stamp reduce end; finish 45
}
for TAG in D0a I0a I1a D1a D1b I1b I0b D0b; do
  run_arm $TAG
  case "$TAG" in D0a|I0a|I1a|D1a) gate_check $TAG;; esac
done
stamp reduce start; say "reduce"
reduce_all || { say "REDUCER FAILED"; finish 22; }
stamp reduce end
V=$(python -c "import json; print(json.load(open('$W/verdict_p223.json'))['verdict'])")
case "$V" in READ) ;; *) say "P223 $V -- no reading (NOISY and VOID are not readings)"; finish 27;; esac
finish 0
