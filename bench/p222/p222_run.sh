#!/bin/bash
# P222's box runner (bench/p222/PREREG-p222.md): K25's decode x plan at Mixtral-8x7B's W16 shapes, a microbench whose
# control is Mixtral's served default (the tree at today's plan). P160's runner (bench/p160/p160_run.sh) by counted
# substitutions -- this lane's run ids, bench, reducer and fit; no router replay (the routing is seeded in the reducer);
# the tripwire reads the tree and today's plan as mixtral's resolved defaults and c1 as e4b's _K25_PLANS["c1"]. P160's pin
# (grouped-nf4-gemm v0.50.0, 41ef086e) and its three kernel files, the power floor (rc 17), refusals, the premise and the
# GPU state per phase are kept. No instance id and no host name are written here. This script never creates compute.
set -uo pipefail
W=/root/p222; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p222: $*"; }
NONCE=${P222_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P222_RUN_NONCE.tmp && mv $W/P222_RUN_NONCE.tmp $W/P222_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P222_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P222_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P222_RUN_ID P222_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949 # grouped-nf4-gemm v0.50.0 (gnf4#572): K39, the K25 lut="prmt"; P158's pin
NF4_SMALLM_SHA256=ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568   # kernel/nf4_smallm.py, K25 + K39
INT4_B32_SHA256=773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f     # kernel/int4_b32.py, P155's
NF4_GROUPED_SHA256=a7090915485c6e9943f568e79315b3f533c754f24edbdb7160b05dcc18619a23  # kernel/nf4_grouped.py: K39's tables
INSTALL_TO_GATE_S=600   # the install (450 s) and the self-tests with the premise (150 s): P152's and P154's walls, rounded up
NEED_BENCH=2100         # the sweep (288 compiles at P156's slowest) and the reading, from P156's and P145's records, x1.5 (PREREG, "Budget")
RESERVE=600             # can_run's margin after the bench: the reduce, the fetch-back and the teardown
MIN_POWER_W=575   # P149's power floor (P156's and P158's, kept): a capped board refuses before the install, rc 17
GPU_CLASS=5090; MIN_RAM_GB=16; MIN_DISK_GB=20
SHAKE=0   # p222-shake-N: the paid shakedown on a cheap card (the A2000 is off lane work since 2026-10-21); never a reading
case "$P222_RUN_ID" in
  p222-5090-1|p222-5090-2|p222-5090-3) :;;
  p222-shake-1|p222-shake-2) SHAKE=1;;
  *) say "unregistered run ID: $P222_RUN_ID"; finish 78;;
esac
[ "$SHAKE" = 1 ] && NEED_BENCH=600   # the shakedown's own fit (PREREG): 2 layers, 6 plans, 8 steps; ~70 s, floored at 600
[ -z "${P222_GPU_CLASS:-}${P222_MIN_DISK_GB:-}${P222_MIN_RAM_GB:-}${P222_MIN_POWER_W:-}${P222_NEED_BENCH:-}" ] || { say 'reading knobs are frozen'; finish 78; }
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
: > summary.txt; echo "$P222_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W need_bench=$NEED_BENCH" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p222_bench.py p222_reduce.py test_k25_row_exact_gpu.py test_nf4_smallm_prmt_gpu.py staged-p222.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p222.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p222/staged-p222.sha256"; finish 9; }
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
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P222_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
gpu_snap(){ local q; q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$q" ] || q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ),$1,${q:-unread}" >> $W/gpu_phases.csv; }
echo "utc,phase,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active" > $W/gpu_phases.csv
# the time-left check: FIT (the step's need plus the reserve) is computed once, compared, and printed as compared
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P222_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P222_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
assert md.version("grouped-nf4-gemm") == "0.50.0", md.version("grouped-nf4-gemm")   # the release carrying K39
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
assert hr._K25_PLAN == {"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}, hr._K25_PLAN
# T is Mixtral's served default: an unset knob resolves mixtral's K25 decode to the tree and its plan to today's (no read
# licenses mixtral), and c1 is e4b's registered pair (P161's default on qwen3_moe), the arm C1 times
os.environ.pop("E4B_K25_LUT", None); os.environ.pop("E4B_K25_PLAN", None)
_r = hr.resolve_k25_lut("mixtral")
assert (_r["env"], _r["lut"]) == (None, "tree") and _r["source"].startswith("default-off"), _r
_p = hr.resolve_k25_plan(_r, model_type="mixtral")
assert (_p["env"], _p["plan"]) == (None, "today"), _p
import sys; sys.path.insert(0, "/root/p222"); import p222_reduce as R
assert hr._K25_PLANS["c1"] == (R.C1["gu"], R.C1["dn"]), hr._K25_PLANS["c1"]
assert {k: v for k, v in hr._K25_PLAN.items() if k in R.TODAY} == R.TODAY, "T is today's served plan"
import torch, triton
import experts4bit_qlora as e
open("/root/p222/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p222_bench.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BENCH SELF-TEST FAILED"; finish 21; }
python $W/p222_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "P222 REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card: K25's row-exact GPU tests (4) and K39's compiled contract (grouped-nf4-gemm v0.50.0's
# kernel/test_nf4_smallm_prmt_gpu.py, 23; P158's premise), the kernel under read
(cd $W && PYTHONPATH='' perl -e 'alarm 600; exec @ARGV' python -m pytest test_k25_row_exact_gpu.py test_nf4_smallm_prmt_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^27 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the microbench: block A's sweep (the selection of M), then block B's five arms (the reading), a spacer before
# every timed replay
can_run $NEED_BENCH bench || finish 40
AL=$(step_alarm 2700); say "microbench (alarm=$AL)"; gpu_snap "bench start"
PYTHONPATH= perl -e "alarm $AL; exec @ARGV" python $W/p222_bench.py --out $W/bench_p222.json $([ "$SHAKE" = 1 ] && echo --shakedown --layers 2 --reps 10 --grid 6 --select-steps 2) > logs/bench.log 2>&1
rc=$?; gpu_snap "bench end"
{ echo -n "bench rc=$rc "; grep -a "^P222_BENCH\|^P222_POISONED\|^P222_ARM_FAULT\|^P222_M \|^P222_SELECT_NONE" logs/bench.log | tail -3 | cut -c1-600; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -4 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED (rc=$rc) -- the reducer will VOID"; }
say "reduce"; python $W/p222_reduce.py --bench "$W/bench_p222.json" --out "$W/verdict_p222.json" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p222.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p222.json'))['verdict'])")
case "$SHAKE/$V" in 0/GO|0/NO_GO|1/SHAKEDOWN) finish 0;; *) say "P222 $V -- no reading"; finish 27;; esac
