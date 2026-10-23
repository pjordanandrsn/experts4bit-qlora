#!/bin/bash
# P228's box runner (bench/p228/PREREG-p228.md): K1's selection and microbench (E4B_DENSE_LT_PIN) on one RTX 5090.
# P227's runner (bench/p227/p227_run.sh) by substitution -- this lane's run ids, bench, reducer, premise and fit; no
# kernel package is installed (the pinned path is cuBLASLt through torch's own library). The power floor (rc 17), the
# refusals and the GPU state per phase are kept. No instance id and no host name are written here. This script never
# creates compute.
set -uo pipefail
W=/root/p228; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p228: $*"; }
NONCE=${P228_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P228_RUN_NONCE.tmp && mv $W/P228_RUN_NONCE.tmp $W/P228_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P228_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P228_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P228_RUN_ID P228_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
INSTALL_TO_GATE_S=600   # the install, the self-tests and the premise: P222's, kept
NEED_BENCH=1200         # the selection (up to 64 candidates a cell) and the reading: about 480 s (PREREG), x2, rounded up
RESERVE=600             # can_run's margin after the bench: the reduce, the fetch-back and the teardown
MIN_POWER_W=575   # P149's power floor (P222's, kept): a capped board refuses before the install, rc 17
GPU_CLASS=5090; MIN_RAM_GB=16; MIN_DISK_GB=20
SHAKE=0   # p228-shake-N: the shakedown on a cheaper card; never a reading
case "$P228_RUN_ID" in
  p228-5090-1|p228-5090-2|p228-5090-3|p228-5090-r1|p228-5090-r2|p228-5090-r3|p228-5090-r4|p228-5090-r5|p228-5090-r6|p228-5090-r7|p228-5090-r8) :;;   # r1..r8: Amendment 5's retry ids (labels, not budget)
  p228-shake-1|p228-shake-2|p228-shake-3) SHAKE=1;;
  *) say "unregistered run ID: $P228_RUN_ID"; finish 78;;
esac
[ "$SHAKE" = 1 ] && NEED_BENCH=600   # the shakedown's own fit: fewer copies, 4 reps a cell; well under 600 s
[ -z "${P228_GPU_CLASS:-}${P228_MIN_DISK_GB:-}${P228_MIN_RAM_GB:-}${P228_MIN_POWER_W:-}${P228_NEED_BENCH:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever, engine knob and kernel knob starts unset: the bench installs the pins itself
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_INT4_WIDE_TILES E4B_INT4_TILE_PROGRAMS \
      E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW \
      GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_INT8_HEAD_BW E4B_K25_LUT E4B_K25_PLAN E4B_NF4_SMALLM_BW
: > summary.txt; echo "$P228_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W need_bench=$NEED_BENCH" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p228_bench.py p228_reduce.py test_dense_lt_pin_gpu.py test_dense_lt_pin.py staged-p228.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p228.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p228/staged-p228.sha256"; finish 9; }
# ---- refusals before anything is installed: CUDA, the card class, the power floor, the disk, the host RAM, the guard fit
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
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE})"; echo "refused: cuda unusable" > REFUSAL; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "$SHAKE" = 1 ]; then   # any NVIDIA card at sm_86 or later with 15 GB: the structure, never the speed
  case "$CC" in 8.6|8.9|9.0|10.0|12.0) ;; *) say "REFUSED: a shakedown needs compute capability 8.6 or later, got $CC"; finish 15;; esac
else
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; finish 15; }
fi
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
VRAM_FLOOR=30000; [ "$SHAKE" = 1 ] && VRAM_FLOOR=15000
[ "${VRAM_MIB:-0}" -ge "$VRAM_FLOOR" ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB below the ${VRAM_FLOOR} MiB floor"; finish 15; }
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
[ -n "$PL" ] || { say "REFUSED: power.limit unreadable"; echo "refused: power.limit unreadable" > REFUSAL; finish 15; }
echo "POWER_LIMIT_W $PL (floor $MIN_POWER_W)" | tee -a summary.txt
[ "$SHAKE" = 1 ] || [ "$PL" -ge "$MIN_POWER_W" ] || { say "REFUSED: power.limit ${PL} W < ${MIN_POWER_W} W (host-limited power cap)"; echo "refused: power.limit ${PL} W" > REFUSAL; finish 17; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P228_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
gpu_snap(){ local q; q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$q" ] || q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ),$1,${q:-unread}" >> $W/gpu_phases.csv; }
echo "utc,phase,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active" > $W/gpu_phases.csv
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P228_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P228_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
TORCH_PIN=2.8.0; echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
WANT_E4B=$E4B_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
import torch, torch.nn.functional as F
from experts4bit_qlora.engines import dense_lt_pin as dl, router_epilogue as re_
assert dl.LT_PINS == {}, "the commit ships no pins: this lane selects them"
assert dl.SITES == ("o_proj", "router") and (dl.MIN_M, dl.MAX_M) == (2, 8) and dl.REDUCTION_OK == (0, 2, 4)
assert re_.router_linear is dl.router_linear, "the router epilogue routes its GEMM through the pin's call"
F.linear(torch.ones(2, 8, device="cuda", dtype=torch.bfloat16), torch.ones(8, 8, device="cuda", dtype=torch.bfloat16))
lib = dl.LtLib(dl.mapped_cublaslt())
import experts4bit_qlora as e
open("/root/p228/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ntorch {torch.__version__}\ncublaslt {lib.version()} {lib.path}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "cublaslt", lib.version())
PYT
cat versions.txt | tee -a summary.txt
python $W/p228_bench.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BENCH SELF-TEST FAILED"; finish 21; }
python $W/p228_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "P228 REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card (the maintainer's condition): K1's GPU suite against the loaded cuBLASLt (10) and its CPU
# suite (27), staged copies; sm_120's heuristic list and kernels differ from any shakedown card's
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_dense_lt_pin_gpu.py test_dense_lt_pin.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^37 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the microbench: per site and M, block A's selection over cuBLASLt's own list, then block B's four arms
can_run $NEED_BENCH bench || finish 40
AL=$(step_alarm 1800); say "microbench (alarm=$AL)"; gpu_snap "bench start"
PYTHONPATH= perl -e "alarm $AL; exec @ARGV" python $W/p228_bench.py --out $W/bench_p228.json $([ "$SHAKE" = 1 ] && echo --shakedown --reps 4 --copies-o 8 --copies-r 32) > logs/bench.log 2>&1
rc=$?; gpu_snap "bench end"
{ echo -n "bench rc=$rc "; grep -a "^P228_BENCH" logs/bench.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -4 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED (rc=$rc) -- the reducer will VOID"; }
say "reduce"; python $W/p228_reduce.py --bench "$W/bench_p228.json" --out "$W/verdict_p228.json" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p228.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p228.json'))['verdict'])")
# a reading finishes 0 only when at least one site was READ: PASS, MISS or TOLERANCE_FAIL (final for that site, never
# re-read). A record whose every site is VOID is not a reading (rc 27). A shakedown finishes 0 only on both SHAKEDOWN.
READ=$(python -c "import json; v=json.load(open('$W/verdict_p228.json')); print(sum(f.get('read') is True for f in (v.get('sites') or {}).values()))")
case "$SHAKE/$V" in
  "1/o_proj=SHAKEDOWN router=SHAKEDOWN") finish 0;;
  1/*) say "P228 shakedown $V -- not a clean shakedown"; finish 27;;
  0/*) [ "${READ:-0}" -ge 1 ] && finish 0; say "P228 $V -- no site read"; finish 27;;
esac
