#!/bin/bash
# P156's box runner (bench/p156/PREREG-p156.md): K39's decode (K25's lut="prmt", grouped-nf4-gemm#569) against
# today's select tree at Qwen3-30B-A3B's W16 shapes, a microbench on S0's real router picks. P155's runner
# (bench/p155/p155_run.sh) by counted substitutions -- this lane's run ids, bench, reducer and fit; the pin,
# grouped-nf4-gemm main at 68bbefc4 (#569's merge, after v0.49.0; K39 unreleased), with K25's and K39's files pinned;
# Nsight on T's and P's calls. P155's power floor (rc 17), refusals, premise, pins and GPU state per phase are kept.
# No instance id and no host name are written here. This script never creates compute.
set -uo pipefail
W=/root/p156; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p156: $*"; }
NONCE=${P156_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P156_RUN_NONCE.tmp && mv $W/P156_RUN_NONCE.tmp $W/P156_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P156_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P156_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P156_RUN_ID P156_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=68bbefc4f43aaa83870903ac2c8788bab3ab1b89 # grouped-nf4-gemm main at 68bbefc4 (#569's merge, after v0.49.0; K39 unreleased); the tree path is P155's
NF4_SMALLM_SHA256=ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568   # kernel/nf4_smallm.py, K25 + K39
INT4_B32_SHA256=773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f     # kernel/int4_b32.py, P155's
NF4_GROUPED_SHA256=a7090915485c6e9943f568e79315b3f533c754f24edbdb7160b05dcc18619a23  # kernel/nf4_grouped.py: K39's tables
INSTALL_TO_GATE_S=600   # the install (450 s) and the self-tests with the premise (150 s): P152's and P154's walls, rounded up
NEED_BENCH=1600         # the microbench: the larger of P155's box walls and the shakedown's, x1.5 (PREREG, "Budget")
NEED_NCU=300            # REPORTED only: Nsight Compute on T's and P's gate_up and down calls (an install, a probe)
RESERVE=600             # can_run's margin after the bench: the reduce, the fetch-back and the teardown
MIN_POWER_W=575   # P149's power floor (P152's and P154's, kept): a capped board refuses before the install, rc 17
GPU_CLASS=5090; MIN_RAM_GB=16; MIN_DISK_GB=20
case "$P156_RUN_ID" in
  p156-5090-1|p156-5090-2|p156-5090-3) :;;
  *) say "unregistered run ID: $P156_RUN_ID"; finish 78;;
esac
[ -z "${P156_GPU_CLASS:-}${P156_MIN_DISK_GB:-}${P156_MIN_RAM_GB:-}${P156_MIN_POWER_W:-}${P156_NEED_BENCH:-}${P156_NEED_NCU:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever, engine knob and kernel knob starts unset: today's served plan is the control
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_INT4_WIDE_TILES E4B_INT4_TILE_PROGRAMS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_INT8_HEAD_BW
: > summary.txt; echo "$P156_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W need_bench=$NEED_BENCH" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p156_bench.py p156_reduce.py rr.json.pt test_k25_row_exact_gpu.py staged-p156.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p156.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p156/staged-p156.sha256"; finish 9; }
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
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + NEED_NCU + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P156_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
gpu_snap(){ local q; q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$q" ] || q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ),$1,${q:-unread}" >> $W/gpu_phases.csv; }
echo "utc,phase,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active" > $W/gpu_phases.csv
# the time-left check: FIT (the step's need plus the reserve) is computed once, compared, and printed as compared
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P156_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P156_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
  "transformers==5.17.0" "bitsandbytes==0.50.2" accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_SMALLM=$NF4_SMALLM_SHA256 WANT_B32=$INT4_B32_SHA256 WANT_GROUPED=$NF4_GROUPED_SHA256 python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, hashlib, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
# main after the v0.49.0 tag: the metadata still reads 0.49.0, and K39 is in no release (the pin is the commit)
assert md.version("grouped-nf4-gemm") == "0.49.0", md.version("grouped-nf4-gemm")
import nf4_smallm, int4_b32, nf4_grouped
for mod, want in ((nf4_smallm, os.environ["WANT_SMALLM"]), (int4_b32, os.environ["WANT_B32"]),
                  (nf4_grouped, os.environ["WANT_GROUPED"])):
    got = hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()
    assert got == want, f"{mod.__name__} is {got}, pinned {want}: not the K25 and K39 this lane registers"
assert nf4_smallm._LUT_MODES.get("prmt") == 4, "K39's decode mode is the kernel's LUT 4"
from experts4bit_qlora.engines import hot_residency as hr
assert hr._k25_mode_env() == "auto", "K25 serves the NF4 store's batched decode rows by default"
assert hr._lean_glue_mode_env() == "auto", "K23's lean glue is the default"
assert hr._lean_glue_supported(int4_b32.build_group_tiles_fused, nf4_smallm.gemm_nf4_grouped_smallm), \
    "the served call carries gather_div / scatter (the bench's call form)"
assert hr._k25_plan(nf4_smallm) == {"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}, \
    "today's plan is the served plan"
import torch, triton
import experts4bit_qlora as e
open("/root/p156/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p156_bench.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BENCH SELF-TEST FAILED"; finish 21; }
python $W/p156_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "P156 REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card: K25's row-exact GPU tests (the kernel under read)
(cd $W && PYTHONPATH='' perl -e 'alarm 600; exec @ARGV' python -m pytest test_k25_row_exact_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^4 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the microbench: identity and the prmt sweep over the grid, then the paired, rotated confirmation
can_run $NEED_BENCH bench || finish 40
AL=$(step_alarm 3600); say "microbench (alarm=$AL)"; gpu_snap "bench start"
PYTHONPATH= perl -e "alarm $AL; exec @ARGV" python $W/p156_bench.py --rr $W/rr.json.pt --out $W/bench_p156.json > logs/bench.log 2>&1
rc=$?; gpu_snap "bench end"
{ echo -n "bench rc=$rc "; grep -a "^P156_BENCH\|^P156_CARRIED\|^P156_POISONED\|^P156_ROUTING_FAULT" logs/bench.log | tail -3 | cut -c1-600; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -4 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED (rc=$rc) -- the reducer will VOID"; }
say "reduce"; python $W/p156_reduce.py --bench "$W/bench_p156.json" --out "$W/verdict_p156.json" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p156.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p156.json'))['verdict'])")
# ---- REPORTED only, after the verdict is written: Nsight Compute on T's and P's gate_up and down calls at today's plan on S0's
# routing (the maintainer's ruling). A probe first: no ncu, a refused counter (ERR_NVGPUCTRPERM is common in rented
# containers) or no time left records why and moves on. It never changes the verdict or the exit code.
ncu_step(){
  local NCU rc
  NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
  if [ -z "$NCU" ]; then
    { perl -e 'alarm 90; exec @ARGV' apt-get update -q && perl -e 'alarm 150; exec @ARGV' apt-get install -y -q cuda-nsight-compute-12-8; } > logs/ncu_install.log 2>&1
    NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
  fi
  [ -n "$NCU" ] || { echo "NCU unavailable: no ncu on this host" | tee -a summary.txt; return 0; }
  PYTHONPATH= perl -e 'alarm 120; exec @ARGV' "$NCU" --metrics sm__cycles_elapsed.avg --kernel-name regex:gemm_nf4_grouped_smallm \
    --launch-count 1 python $W/p156_bench.py --rr $W/rr.json.pt --ncu-target > logs/ncu_probe.log 2>&1
  if grep -q "ERR_NVGPUCTRPERM" logs/ncu_probe.log || ! grep -q "sm__cycles_elapsed.avg" logs/ncu_probe.log; then
    echo "NCU counters unavailable on this host ($(grep -m1 -o 'ERR_[A-Z_]*' logs/ncu_probe.log || echo 'no metric line'))" | tee -a summary.txt
    return 0
  fi
  PYTHONPATH= perl -e 'alarm 240; exec @ARGV' "$NCU" --clock-control none --kernel-name regex:gemm_nf4_grouped_smallm \
    --launch-skip 12 --launch-count 4 --section SpeedOfLight --section Occupancy --section LaunchStats \
    --section WarpStateStats --section MemoryWorkloadAnalysis --csv --page details \
    python $W/p156_bench.py --rr $W/rr.json.pt --ncu-target > ncu_p156.csv 2> logs/ncu.log
  rc=$?; echo "NCU captured rc=$rc ($(wc -l < ncu_p156.csv) csv lines, $("$NCU" --version 2>/dev/null | tail -1))" | tee -a summary.txt
}
if can_run $NEED_NCU "ncu (reported only)"; then gpu_snap "ncu start"; ncu_step; gpu_snap "ncu end"; fi
case "$V" in GO|NO_GO) finish 0;; *) say "P156 $V -- no reading"; finish 27;; esac
