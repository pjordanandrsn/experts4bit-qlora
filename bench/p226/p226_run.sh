#!/bin/bash
# P226's box runner (bench/p226/PREREG-p226.md): P154's runner (bench/p154/p154_run.sh) by named edits. One box: the
# checkpoint and its NF4 arena; the ROUTING CAPTURE of 32 distinct requests at bucket 32 (p226_capture.py over P154's
# routing pass: graphs off so the router hooks fire, max_seqs 32, the served default otherwise, the K25 route and the c1
# plan asserted over the window, the picks hashed before any timing); then the MICROBENCH on those picks (p226_bench.py:
# P156's sweep, reported, and P160's instrument, the ruling) and the reducer. grouped-nf4-gemm is pinned at v0.50.0, the
# release that carries K39's prmt decode (qwen3_moe's served decode and c1's premise). P149's power floor (rc 17), the
# refusals, pins and GPU state per phase are kept; no dry mode. This script never creates compute.
set -uo pipefail
W=/root/p226; mkdir -p $W/logs $W/run; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p226: $*"; }
NONCE=${P226_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P226_RUN_NONCE.tmp && mv $W/P226_RUN_NONCE.tmp $W/P226_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P226_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P226_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P226_RUN_ID P226_DEADLINE_EPOCH P226_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
STARTED_AT=$(date +%s)   # the arena must be written after this (the box's own bake, not a shipped file)
GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949 # grouped-nf4-gemm v0.50.0 (gnf4#572): K39's prmt decode; P160's pin
# the needs (PREREG "Cost"): the slowest measured fetch (806 s) and bake (362 s) x 1.5; the capture (load, 32 prefills,
# 160+ eager steps with hooks); the bench runs P156's selection AND P160's confirmation: (462 + 260) s at 16 rows, about
# 1,444 s at 32, x 1.50
NEED_FETCH=1210; NEED_BAKE=550; NEED_CAPTURE=900; NEED_BENCH=2170
INSTALL_TO_GATE_S=450   # the two installs and the self-tests (P224's allowance)
RESERVE=600             # can_run's margin after a step: the reduce, the fetch-back and the teardown
MIN_POWER_W=575   # P149's power floor (P152's, kept): a capped board refuses before the install, rc 17
GPU_CLASS=5090; MIN_RAM_GB=60
case "$P226_RUN_ID" in
  p226-5090-1|p226-5090-2|p226-5090-3) FAMILY=qwen3; MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; MOE_LAYERS=48; MIN_DISK_GB=150;;
  *) say "unregistered run ID: $P226_RUN_ID"; finish 78;;
esac
[ -z "${P226_GPU_CLASS:-}${P226_MIN_DISK_GB:-}${P226_MIN_RAM_GB:-}${P226_MIN_POWER_W:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_INT8_HEAD_BW
: > summary.txt; echo "$P226_INSTANCE_ID" > INSTANCE_ID; echo "$P226_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB family=$FAMILY min_power_w=$MIN_POWER_W moe_layers=$MOE_LAYERS need_bench=$NEED_BENCH" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p226_capture.py p226_bench.py p226_reduce.py p154_route.py p136_box.py sc1b_e4b_census.py p109_box.py k8_bake.py \
         calib.json test_k25_row_exact_gpu.py staged-p226.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p226.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p226/staged-p226.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the card class, the power floor, the disk, the host RAM
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
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: 5090 VRAM ${VRAM_MIB:-?} MiB below 30000 MiB floor"; finish 15; }
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
[ -n "$PL" ] || { say "REFUSED: power.limit unreadable"; echo "refused: power.limit unreadable" > REFUSAL; finish 15; }
echo "POWER_LIMIT_W $PL (floor $MIN_POWER_W)" | tee -a summary.txt
[ "$PL" -ge "$MIN_POWER_W" ] || { say "REFUSED: power.limit ${PL} W < ${MIN_POWER_W} W (host-limited power cap)"; echo "refused: power.limit ${PL} W" > REFUSAL; finish 17; }
gpu_snap(){ local q; q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$q" ] || q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ),$1,${q:-unread}" >> $W/gpu_phases.csv; }
echo "utc,phase,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active" > $W/gpu_phases.csv
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the checkpoint, the NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
# ---- the pre-install fit check (bench/common/README.md, "Guard fit"): before any install and any staged script
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_FETCH + NEED_BAKE + NEED_CAPTURE + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P226_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
# the time-left check: FIT (the step's need plus the 600 s reserve) is computed once, compared, and printed as compared
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + 600)); left=$((P226_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the 600 s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P226_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
TORCH_PIN=2.8.0; echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == "0.50.0", md.version("grouped-nf4-gemm")   # the release carrying K39
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged as sp
assert sp._graphs_env("", "cuda", "all-vram") is True, "the served default captures graphs (the route under read is the replay's)"
from experts4bit_qlora.engines import hot_residency as hr
assert hr._k25_mode_env() == "auto", "K25 serves the NF4 store's batched decode rows by default"
hr.DEVICE_GROUPING[0] = True
assert hr._collapsed_grouping(32, None) == (False, True), "DEVICE_GROUPING on: 32 rows take the device-grouped route"
hr.DEVICE_GROUPING[0] = False
lut = hr.resolve_k25_lut("qwen3_moe")
assert lut.get("lut") == "prmt", f"qwen3_moe's served K25 decode is prmt (P158): {lut}"
plan = hr.resolve_k25_plan(lut, True, "qwen3_moe")
assert plan.get("plan") == "c1", f"qwen3_moe's served K25 plan is c1 (P161): {plan}"
assert hr._K25_PLANS["c1"] == ({"block_n": 64, "kc": 256, "warps": 4, "stages": 2},
                               {"block_n": 128, "kc": 128, "warps": 8, "stages": 3}), "c1 is P160's C1"
import nf4_smallm
assert callable(nf4_smallm.gemm_nf4_grouped_smallm), "K25 in the kernel package"
import torch, triton
import experts4bit_qlora as e
open("/root/p226/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p154_route.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "ROUTE SELF-TEST FAILED"; finish 21; }
for t in p226_capture p226_bench p226_reduce; do
  python $W/$t.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "$t SELF-TEST FAILED"; finish 21; }
done
python $W/p136_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/sc1b_e4b_census.py --selftest | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "WINDOW SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: K25's row-exact GPU tests (the kernel the route under read uses)
(cd $W && PYTHONPATH='' perl -e 'alarm 600; exec @ARGV' python -m pytest test_k25_row_exact_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^4 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as SC1 bakes it) and the prompts
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED before fetch: disk ${FREE_GB:-?} GB"; finish 13; }
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', ignore_patterns=['original/*','metal/*','consolidated*'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p226_capture.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts_p226.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P226_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the ROUTING CAPTURE: 32 distinct requests (wikitext rows 16..47) at bucket 32 on the served default (the int8 head
# included), eager so the router hooks fire, with DEVICE_GROUPING set by the pass; the K25 route and the c1 plan are asserted
# over the 128-step window, and the picks are hashed into capture.json BEFORE any timing
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=32"
can_run $NEED_CAPTURE "routing capture" || finish 40
AL=$(step_alarm 1800); say "routing capture B=32 (alarm=$AL)"; gpu_snap "capture start"
# shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
env PYTHONPATH= $ENGINE_ENV E4B_PAGED_GRAPHS=0 E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P226_STARTED_AT=$STARTED_AT \
  perl -e "alarm $AL; exec @ARGV" python $W/p226_capture.py --prompts $W/prompts_p226.json --out $W/run/capture.json \
    --picks $W/run/capture.pt > logs/capture.log 2>&1
rc=$?; gpu_snap "capture end"
{ echo -n "capture rc=$rc "; grep -a "^P226_CAPTURE" logs/capture.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
if [ "$rc" = 0 ] && [ -s $W/run/capture.pt ]; then
  # ---- the MICROBENCH on the captured picks: the sweep (reported), then the confirmation (the ruling)
  can_run $NEED_BENCH bench || finish 40
  AL=$(step_alarm 4800); say "bench (alarm=$AL)"; gpu_snap "bench start"
  env PYTHONPATH= perl -e "alarm $AL; exec @ARGV" python $W/p226_bench.py --picks $W/run/capture.pt --out $W/run/bench_p226.json > logs/bench.log 2>&1
  rc=$?; gpu_snap "bench end"
  { echo -n "bench rc=$rc "; grep -a "^P226_CARRIED\|^P226_BENCH" logs/bench.log | cut -c1-600; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -4 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED (rc=$rc) -- the reducer will VOID"; }
else
  tail -4 logs/capture.log | cut -c1-300 | tee -a summary.txt; say "CAPTURE FAILED OR FAULTED (rc=$rc) -- no bench; the reducer will VOID"
fi
say "reduce"; python $W/p226_reduce.py --bench "$W/run/bench_p226.json" --capture "$W/run/capture.json" --out "$W/run/verdict_p226.json" 2>&1 | tee -a summary.txt
[ -s "$W/run/verdict_p226.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/run/verdict_p226.json'))['verdict'])")
case "$V" in GO|NO_GO) finish 0;; *) say "P226 $V -- no reading"; finish 27;; esac
