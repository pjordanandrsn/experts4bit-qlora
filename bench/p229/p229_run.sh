#!/bin/bash
# P229's box runner (bench/p229/PREREG-p229.md): K2b's microbench (E4B_INT8_HEAD_ROWS) on one RTX 5090.
# P229's runner (bench/p229/p229_run.sh) by substitution -- this lane's run ids, bench, reducer, premise and fit, plus
# grouped-nf4-gemm installed at its registered release commit (K36 and K37 are its kernels). The power floor (rc 17),
# the refusals and the GPU state per phase are kept. No instance id and no host name are written here. This script
# never creates compute.
set -uo pipefail
W=/root/p229; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p229: $*"; }
NONCE=${P229_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P229_RUN_NONCE.tmp && mv $W/P229_RUN_NONCE.tmp $W/P229_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P229_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P229_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P229_RUN_ID P229_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
INSTALL_TO_GATE_S=900   # two installs (grouped-nf4-gemm, then e4b), the self-tests and the two-part premise
NEED_BENCH=600          # the reading and its Triton compiles: about 120 s (PREREG), x2, rounded up to the floor
RESERVE=600             # can_run's margin after the bench: the reduce, the fetch-back and the teardown
MIN_POWER_W=575   # P149's power floor (P222's, kept): a capped board refuses before the install, rc 17
GPU_CLASS=5090; MIN_RAM_GB=16; MIN_DISK_GB=20
SHAKE=0   # p229-shake-N: the shakedown on a cheaper card; never a reading
case "$P229_RUN_ID" in
  p229-5090-1|p229-5090-2|p229-5090-3|p229-5090-r1|p229-5090-r2|p229-5090-r3|p229-5090-r4|p229-5090-r5|p229-5090-r6|p229-5090-r7|p229-5090-r8) :;;   # r1..r8: Amendment 4's retry ids (labels, not budget)
  p229-shake-1|p229-shake-2|p229-shake-3) SHAKE=1;;
  *) say "unregistered run ID: $P229_RUN_ID"; finish 78;;
esac
GNF4=41ef086e6a3e2be5b158ff1bb38e0de67862a949   # grouped-nf4-gemm v0.50.0, released: K36 and K37 as shipped
[ -z "${P229_GPU_CLASS:-}${P229_MIN_DISK_GB:-}${P229_MIN_RAM_GB:-}${P229_MIN_POWER_W:-}${P229_NEED_BENCH:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever, engine knob and kernel knob starts unset: the bench installs the route itself
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_INT4_WIDE_TILES E4B_INT4_TILE_PROGRAMS \
      E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW \
      GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_INT8_HEAD_BW E4B_K25_LUT E4B_K25_PLAN E4B_NF4_SMALLM_BW \
      E4B_INT8_HEAD_ROWS E4B_PAGED_SPEC E4B_PAGED_SPEC_K E4B_DENSE_LT_PIN
: > summary.txt; echo "$P229_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4 gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W need_bench=$NEED_BENCH" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p229_bench.py p229_reduce.py staged-p229.sha256 t/tests/test_int8_head_bw.py t/tests/test_int8_head_rows_interp.py \
         t/tests/conftest.py t/bench/i8dense/i8_fakequant.py t/bench/p148/p148_kernels.py; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p229.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p229/staged-p229.sha256"; finish 9; }
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
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P229_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
gpu_snap(){ local q; q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$q" ] || q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ),$1,${q:-unread}" >> $W/gpu_phases.csv; }
echo "utc,phase,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active" > $W/gpu_phases.csv
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P229_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P229_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
TORCH_PIN=2.8.0; echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install grouped-nf4-gemm @$GNF4 (no deps), then e4b @$E4B_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_gnf4.log 900 --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4" || { tail -4 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4 python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
g = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert g.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed grouped-nf4-gemm is not the release commit: {g}"
import torch, triton, dense_i8, dense_gemv
from experts4bit_qlora.engines import int8_head_bw as ih
assert ih.MAX_ROWS == 8 and ih.ROWS_ENV == "E4B_INT8_HEAD_ROWS" and callable(ih.apply_int8_head_rows)
assert all(tuple(dense_i8.plan(151936, 2048, m)) == (32, 256, 4) for m in range(2, 9)), "the head's shipped plan per M"
import experts4bit_qlora as e
open("/root/p229/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngrouped-nf4-gemm {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"))
PYT
cat versions.txt | tee -a summary.txt
python $W/p229_bench.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BENCH SELF-TEST FAILED"; finish 21; }
python $W/p229_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "P229 REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card: K2b's route through the REAL kernels compiled here (the interpreter file with
# TRITON_INTERPRET=0: 11, its four CUDA-graph replays included) and its CPU suite (45, whose wrapper runs the same file
# interpreted), staged copies in the repo's own layout under t/ (the CPU suite reads bench/ files beside tests/)
(cd $W/t && PYTHONPATH='' TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest tests/test_int8_head_rows_interp.py -q -rs -p no:cacheprovider) > logs/premise_gpu.log 2>&1
rc1=$?; L1=$(tail -1 logs/premise_gpu.log)
(cd $W/t && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest tests/test_int8_head_bw.py -q -rs -p no:cacheprovider) > logs/premise_cpu.log 2>&1
rc2=$?; L2=$(tail -1 logs/premise_cpu.log)
{ echo "premise gpu rc=$rc1: $L1"; echo "premise cpu rc=$rc2: $L2"; } | tee -a summary.txt
if [ "$rc1" = 0 ] && echo "$L1" | grep -qE '^11 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$' \
   && [ "$rc2" = 0 ] && echo "$L2" | grep -qE '^45 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc1/$rc2" > REFUSAL; finish 25
fi
# ---- the microbench: per M, the four arms in Williams order behind the spacer
can_run $NEED_BENCH bench || finish 40
AL=$(step_alarm 1800); say "microbench (alarm=$AL)"; gpu_snap "bench start"
PYTHONPATH= perl -e "alarm $AL; exec @ARGV" python $W/p229_bench.py --out $W/bench_p229.json $([ "$SHAKE" = 1 ] && echo --shakedown --reps 4) > logs/bench.log 2>&1
rc=$?; gpu_snap "bench end"
{ echo -n "bench rc=$rc "; grep -a "^P229_BENCH" logs/bench.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -4 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED (rc=$rc) -- the reducer will VOID"; }
say "reduce"; python $W/p229_reduce.py --bench "$W/bench_p229.json" --out "$W/verdict_p229.json" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p229.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p229.json'))['verdict'])")
# a reading finishes 0 only when it was READ: GO_ROWS_8/4/2, NO_GO or TOLERANCE_FAIL (final, never re-read). VOID is not
# a reading (rc 27). A shakedown finishes 0 only on SHAKEDOWN.
case "$SHAKE/$V" in
  1/SHAKEDOWN) finish 0;;
  1/*) say "P229 shakedown $V -- not a clean shakedown"; finish 27;;
  0/GO_ROWS_8|0/GO_ROWS_4|0/GO_ROWS_2|0/NO_GO|0/TOLERANCE_FAIL) finish 0;;
  0/*) say "P229 $V -- not a reading"; finish 27;;
esac
