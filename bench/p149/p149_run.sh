#!/bin/bash
# Lane P149, BOX Q+S (bench/p149/PREREG-p149.md, "Box Q+S" and Amendment 4): int8 weight-only attention projections on
# the Mixtral-8x7B server (E4B_SERVE_ATTN_INT8, grouped-nf4-gemm K37), quality at decode then the served read, on ONE RTX
# 5090 at >= 575 W. the P143 Amendment 3 runner (bench/p143/p143a3_run.sh) by counted edits, with the box K host floors:
#   floors   CUDA 18, the card class 15, the disk 13, host RAM 16, power.limit >= 575 W 17 (unreadable 15), all before the
#            install; then the guard fit over every term below at its allowance (rc 19), also before the install;
#   premise  on THIS card: the grouped-nf4-gemm K37 suite compiled (28 passed) and the e4b int8 kernel path compiled (4 passed),
#            rc 25; no re-timing of box K;
#   order    install + tripwire; reducer and box self-tests; premise; fetch; NF4 arena bake; prompts; the P147 windows; then
#            S1w and its early gates (Amendment 5: build rc 43, instrument rc 44), S0a (its build gate, rc 45), S1a
#            (speed, then Q in the same process for both), the Q reduce (a QUALITY_FAIL stops the box here), T1, S1b,
#            S0b, S0c, S1c, T2, S1d, S0d (the Amendment 4 token-pinned processes in the rotation), the two SANE phases
#            (reported), and the reduce. Every phase stamps its UTC start and end in phases.jsonl.
#   dry      Amendment 5 (P149_DRY=1, run ids p149-dry-2 to -4): shards 1 and 19 with the config and tokenizer, cut to
#            one layer (p149_slice.py), then the FULL rotation at its own allowances; gates and the Q reduce report and
#            go on; the verdict is DRY, rc 0 with no instrument fault, else 29. An instrument run: never a reading, never
#            a strike.
# Refusals: as above. Only 13 and 18 name the machine for the next launcher draw. This script never creates compute.
set -uo pipefail
W=/root/p149; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p149: $*"; }
NONCE=${P149_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P149_RUN_NONCE.tmp && mv $W/P149_RUN_NONCE.tmp $W/P149_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P149_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P149_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P149_RUN_ID P149_DEADLINE_EPOCH P149_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7 # v0.49.0 tag commit: the release that ships K37 (dense_i8)
GNF4_VERSION=0.49.0
SHORT=32; LONG=160; REPS=5; CONT=128; SANE_WINDOWS=12
# the Amendment 4 table: the allowances every time-left check uses (s)
INSTALL=450; NEED_FETCH=3000; NEED_BAKE=1500; NEED_ARM=600; NEED_Q=1200; NEED_REDUCE=120; NEED_TF=840; NEED_SANE=900
NEED_GATE=600   # Amendment 5: S1w, a load and one Q pass per gate
RESERVE=600
GPU_CLASS=5090; MIN_RAM_GB=60; MIN_DISK_GB=200; MIN_POWER_W=575
case "$P149_RUN_ID" in
  p149-5090-1|p149-5090-2|p149-5090-3) DRY=0;;
  p149-dry-2|p149-dry-3|p149-dry-4) DRY=1;;   # Amendment 5: instrument runs on a 1-layer cut, never a reading
  *) say "unregistered run ID: $P149_RUN_ID"; finish 78;;
esac
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; REV=eba92302a2861cdc0098cc54bc9f17cb2c47eb61   # the P141 lock
[ "${P149_DRY:-0}" = "$DRY" ] || { say "refusing: P149_DRY=${P149_DRY:-0} does not match the run id $P149_RUN_ID"; finish 78; }
if [ "$DRY" = 1 ]; then   # the dry table (Amendment 5): two shards, one layer
  NEED_FETCH=900; NEED_BAKE=600; NEED_ARM=240; NEED_Q=480; NEED_TF=240; NEED_SANE=240; NEED_GATE=240; MIN_DISK_GB=60
fi
RD=""; [ "$DRY" = 1 ] && RD=--dry
[ -z "${P149_GPU_CLASS:-}${P149_MIN_DISK_GB:-}${P149_MIN_RAM_GB:-}${P149_REHEARSAL:-}${P149_SHORT:-}${P149_LONG:-}${P149_REPS:-}${P149_MIN_POWER_W:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject (the arms set the lever alone)
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
: > summary.txt; : > phases.jsonl; echo "$P149_INSTANCE_ID" > INSTANCE_ID; echo "$P149_RUN_ID" > RUN_ID
echo "KNOBS run=$P149_RUN_ID dry=$DRY e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W short=$SHORT long=$LONG reps=$REPS" | tee -a summary.txt
stamp(){ printf '{"phase": "%s", "event": "%s", "t": %s}\n' "$1" "$2" "$(date +%s)" >> phases.jsonl; }
# ---- staged pieces, byte-for-byte
for f in p149_box.py p149_reduce.py p149_slice.py p115_quality.py p109_box.py p110_box.py p108_box.py p97_box.py p125_box.py \
         k8_bake.py calib.json test_dense_i8.py test_int8_attn_interp.py staged-p149.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p149.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p149/staged-p149.sha256"; finish 9; }
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
[ -n "$PL" ] || { say "REFUSED: power.limit unreadable"; echo "refused: power.limit unreadable" > REFUSAL; finish 15; }
echo "POWER_LIMIT_W $PL (floor $MIN_POWER_W)" | tee -a summary.txt
[ "$PL" -ge "$MIN_POWER_W" ] || { say "REFUSED: power.limit ${PL} W < ${MIN_POWER_W} W (host-limited power cap)"; echo "refused: power.limit ${PL} W" > REFUSAL; finish 17; }
# the Amendment 4 fit, before the install, in the cross-lane form (bench/common/README.md, "Guard fit"): everything from here
# to the verdict at its allowance, plus the reserve, with 25 % headroom
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL + NEED_FETCH + NEED_BAKE + NEED_GATE + 8 * NEED_ARM + 2 * NEED_Q + NEED_REDUCE + 2 * NEED_TF + 2 * NEED_SANE + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P149_DEADLINE_EPOCH - $(date +%s)))
echo "GUARD_FIT ${FIT}s x ${FIT_HEADROOM_PCT}% = ${FIT_NEED}s of ${LEFT}s left" | tee -a summary.txt
# rc 19: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: guard fit ${LEFT}s < ${FIT_NEED}s" > REFUSAL; finish 19; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + RESERVE)) -le "$P149_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s + ${RESERVE}s reserve = $((need + RESERVE))s, only $((P149_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P149_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == os.environ["WANT_GNF4_VERSION"], md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import triton
assert triton.__version__ == "3.4.0", triton.__version__
# the lever: K37's entry point, e4b's engine, the knob in the snapshot, the census for mixtral
from dense_i8 import linear_i8, quantize_rows_int8  # noqa: F401
from experts4bit_qlora.engines import int8_attn
assert int8_attn.CENSUS == {"mixtral": ("q_proj", "k_proj", "v_proj", "o_proj")}, int8_attn.CENSUS
from experts4bit_qlora import serve_paged as sp
assert "E4B_SERVE_ATTN_INT8" in sp.LEVER_ENV and hasattr(sp, "_attn_int8_census")
assert sp._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
m, s = sp.resolve_fusion_modes({k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}, "mixtral")
assert set(m.values()) == {"auto"} and set(s.values()) == {"default-allowlisted"}, (m, s)
import torch
import experts4bit_qlora as e
open("/root/p149/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p149_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p149_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/p115_quality.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "QUALITY SELF-TEST FAILED"; finish 21; }
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
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as P145 bakes it), the prompts and P147's windows
can_run $NEED_FETCH fetch || finish 40
stamp fetch start; say "fetch $MODEL @ $REV"
PATTERNS="ignore_patterns=['original/*','metal/*','consolidated*']"
# the dry run: shards 1 and 19 hold the embedding, layer 0, the final norm and the head
[ "$DRY" = 1 ] && PATTERNS="allow_patterns=['model-00001-of-00019.safetensors','model-00019-of-00019.safetensors','*.json','tokenizer*']"
perl -e "alarm $(step_alarm $NEED_FETCH); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', $PATTERNS, max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt; stamp fetch end
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
if [ "$DRY" = 1 ]; then
  python $W/p149_slice.py --snapshot "$SNAP" --out $W/slice --layers 1 > logs/slice.log 2>&1 || { tail -3 logs/slice.log; say "SLICE FAIL"; finish 11; }
  grep -a "^SLICE" logs/slice.log | tee -a summary.txt; SNAP=$W/slice
fi
can_run $NEED_BAKE bake || finish 40
stamp bake start; say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm $NEED_BAKE); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt; stamp bake end
perl -e "alarm 1200; exec @ARGV" python $W/p149_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p149_box.py --windows-only --model "$MODEL" --revision "$REV" --out $W/windows.json > logs/windows.log 2>&1 \
  || { tail -4 logs/windows.log; say "WINDOWS FAIL (not P147's)"; finish 19; }
grep -a "^P149_WINDOWS" logs/windows.log | tee -a summary.txt
ENGINE_MODEL="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV"; [ "$DRY" = 1 ] && ENGINE_MODEL="E4B_PAGED_MODEL=$W/slice"
ENGINE_ENV="$ENGINE_MODEL E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_BUCKETS=1,2,4,8,16"
run_arm(){   # TAG [--q]: one speed process (S0 unset, S1 the lever), five timed reps; S0a and S1a then run Q
  local TAG=$1 Q=${2:-} ARM=${1:0:2} LV="" AL need=$NEED_ARM
  [ "$ARM" = S1 ] && LV="E4B_SERVE_ATTN_INT8=1"
  [ -n "$Q" ] && need=$((NEED_ARM + NEED_Q))
  can_run $need "arm $TAG" || finish 40
  AL=$(step_alarm $((need + 600))); say "arm $TAG $Q (alarm=$AL)"; stamp "$TAG" start
  # shellcheck disable=SC2086  # ENGINE_ENV and LV are assignment lists by design
  env PYTHONPATH= $ENGINE_ENV $LV P149_ARM=$ARM E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p149_box.py --mode speed --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS $Q ${Q:+--windows $W/windows.json} > logs/arm_$TAG.log 2>&1
  local rc=$?; stamp "$TAG" end
  { echo -n "arm $TAG rc=$rc "; grep -a "^P149_ARM\|^P149_Q" logs/arm_$TAG.log | tail -3 | cut -c1-400 | tr '\n' ' '; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
}
run_tf(){    # TAG: Amendment 4's token-pinned S0 process on S0a's and S1a's recorded W16 tokens
  local TAG=$1 AL
  can_run $NEED_TF "token-pinned $TAG" || finish 40
  AL=$(step_alarm $((NEED_TF + 600))); say "token-pinned $TAG (alarm=$AL)"; stamp "$TAG" start
  # shellcheck disable=SC2086
  env PYTHONPATH= $ENGINE_ENV P149_ARM=S0 E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p149_box.py --mode tf --tag $TAG --prompts $W/prompts.json \
      --stream-a $W/arm_S0a.json --stream-b $W/arm_S1a.json --out $W/tf_$TAG.json --short $SHORT --long $LONG --reps $REPS > logs/tf_$TAG.log 2>&1
  local rc=$?; stamp "$TAG" end
  { echo -n "token-pinned $TAG rc=$rc "; grep -a "^P149_TF" logs/tf_$TAG.log | cut -c1-200 | tr '\n' ' '; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/tf_$TAG.log | cut -c1-300 | tee -a summary.txt; say "TOKEN-PINNED $TAG FAILED (rc=$rc) -- the qualifier reads QUALIFIER_VOID"; }
}
# ---- Amendment 5, the EARLY GATES, before any scored arm: S1w builds S1 as every S1 arm builds and runs one Q pass per
# gate (never scored); its build (rc 43) and its instrument (rc 44) are read from its record, and S0a build (rc 45)
# before S1a. Each is an instrument refusal, not a strike; the same gate firing on two consecutive runs counts as one
# strike. A dry run reports a gate and goes on, so one dry finds every fault.
gate_check(){   # RECORD PART ARM RC
  python $W/p149_reduce.py --gate "$1" --gate-part "$2" --gate-arm "$3" --e4b-sha "$E4B_SHA" $RD 2>&1 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] && return 0
  if [ "$DRY" = 1 ]; then echo "DRY_GATE_FAULT rc=$4" | tee -a summary.txt; GATE_FAULT=${GATE_FAULT:-$4}; return 0; fi
  say "EARLY GATE ($2, $3): not the registered closed forms -- stopping"; echo "refused: early gate $2 $3" > REFUSAL; finish "$4"
}
can_run $NEED_GATE "gate S1w" || finish 40
AL=$(step_alarm $((NEED_GATE + 600))); say "gate S1w (alarm=$AL)"; stamp S1w start
# shellcheck disable=SC2086
env PYTHONPATH= $ENGINE_ENV E4B_SERVE_ATTN_INT8=1 P149_ARM=S1 E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
  perl -e "alarm $AL; exec @ARGV" python $W/p149_box.py --mode gate --windows $W/windows.json --out $W/gate_S1w.json --tag S1w > logs/gate_S1w.log 2>&1
rc=$?; stamp S1w end
{ echo -n "gate S1w rc=$rc "; grep -a "^P149_GATE_RECORD\|^P149_Q" logs/gate_S1w.log | cut -c1-400 | tr '\n' ' '; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || tail -6 logs/gate_S1w.log | cut -c1-300 | tee -a summary.txt
gate_check $W/gate_S1w.json build S1 43
gate_check $W/gate_S1w.json instrument S1 44
run_arm S0a --q
gate_check $W/arm_S0a.json build S0 45
run_arm S1a --q
# ---- the Q reduce: a QUALITY_FAIL stops the box here (the six remaining processes are not run); a dry run goes on
stamp q_reduce start
python $W/p149_reduce.py --dir $W --out $W/q_verdict.json --e4b-sha "$E4B_SHA" --q-only $RD 2>&1 | tee -a summary.txt
QV=$(python -c "import json; print(json.load(open('$W/q_verdict.json'))['verdict'])" 2>/dev/null); stamp q_reduce end
if [ "$DRY" = 1 ]; then
  echo "DRY: the Q reduce read ${QV:-nothing} on the slice; a dry run goes on" | tee -a summary.txt
else
  case "$QV" in
    GO) ;;
    QUALITY_FAIL) say "Q: QUALITY_FAIL -- a PARK of the lever on this path; the box stops"
                  python $W/p149_reduce.py --dir $W --out $W/verdict_p149.json --e4b-sha "$E4B_SHA" 2>&1 | tee -a summary.txt; finish 0;;
    *) say "Q reduce read ${QV:-nothing} -- no reading"; python $W/p149_reduce.py --dir $W --out $W/verdict_p149.json --e4b-sha "$E4B_SHA" 2>&1 | tee -a summary.txt; finish 27;;
  esac
fi
run_tf T1
run_arm S1b; run_arm S0b; run_arm S0c; run_arm S1c
run_tf T2
run_arm S1d; run_arm S0d
# ---- the SANE phases, reported: S0 then S1, each a fresh process on the server built eager, one window per pass
mkdir -p $W/work/ref
for PH in S0 S1; do
  LV=""; [ "$PH" = S1 ] && LV="E4B_SERVE_ATTN_INT8=1"
  can_run $NEED_SANE "sane $PH" || break
  AL=$(step_alarm $((NEED_SANE + 600))); say "sane $PH (alarm=$AL)"; stamp "sane_$PH" start
  # shellcheck disable=SC2086
  env PYTHONPATH= $ENGINE_ENV E4B_PAGED_GRAPHS=0 $LV P149_ARM=$PH E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p149_box.py --mode sane --out $W/sane_$PH.json --ref-dir $W/work/ref \
      --sane-windows $SANE_WINDOWS --cont $CONT > logs/sane_$PH.log 2>&1
  rc=$?; stamp "sane_$PH" end
  { echo -n "sane $PH rc=$rc "; grep -a "^P149_SANE" logs/sane_$PH.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -4 logs/sane_$PH.log | cut -c1-300 | tee -a summary.txt; say "SANE $PH FAILED (rc=$rc) -- reported only"; break; }
done
stamp reduce start; say "reduce"
python $W/p149_reduce.py --dir $W --out $W/verdict_p149.json --e4b-sha "$E4B_SHA" $RD 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p149.json" ] || { say "REDUCER FAILED"; finish 22; }
stamp reduce end
V=$(python -c "import json; print(json.load(open('$W/verdict_p149.json'))['verdict'])")
if [ "$DRY" = 1 ]; then   # never a reading: rc 0 only when no gate fired and no instrument fault was found
  OK=$(python -c "import json; print(json.load(open('$W/verdict_p149.json'))['ok'])")
  if [ -z "${GATE_FAULT:-}" ] && [ "$OK" = True ]; then say "P149 DRY: no instrument fault"; finish 0; fi
  say "P149 DRY: instrument faults (first gate rc ${GATE_FAULT:-none}) -- fix and re-dry"; finish 29
fi
case "$V" in FLIP|NO_FLIP|FUNCTION_FAIL|QUALITY_FAIL) ;; *) say "P149 $V -- no reading (NOISY and VOID are not readings)"; finish 27;; esac
finish 0
