#!/bin/bash
# P227's box runner (bench/p227/PREREG-p227.md): the serving roadmap's lever B in a microbench, two families. P222's
# runner (bench/p222/p222_run.sh) by substitution -- this lane's run ids, bench, reducer, premise and fit; the routing is
# P153's router replay (staged) and P222's seed. The pin (grouped-nf4-gemm v0.50.0, 41ef086e) and its three kernel files,
# the power floor (rc 17), the refusals and the GPU state per phase are kept. No instance id and no host name are
# written here. This script never creates compute.
set -uo pipefail
W=/root/p227; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p227: $*"; }
NONCE=${P227_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P227_RUN_NONCE.tmp && mv $W/P227_RUN_NONCE.tmp $W/P227_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P227_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P227_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P227_RUN_ID P227_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949 # grouped-nf4-gemm v0.50.0 (gnf4#572): K33, K25 with K39's decode; P222's pin
NF4_SMALLM_SHA256=ee46969a4f786c36abe26bfaf0a52e426db4465406a5e31c6085fd3270b26568   # kernel/nf4_smallm.py, K25 + K39
INT4_B32_SHA256=773eff5db7be77dea3d1f9048a6e708a94e79634baf17f452bf2848e24a2072f     # kernel/int4_b32.py, K23's tile builder
NF4_GROUPED_SHA256=a7090915485c6e9943f568e79315b3f533c754f24edbdb7160b05dcc18619a23  # kernel/nf4_grouped.py: K33
INSTALL_TO_GATE_S=600   # the install, the self-tests and the premise: P222's, kept
NEED_BENCH=1500         # the reading: about 820 s by the PREREG arithmetic, x1.5, rounded up (PREREG, "Budget and the fit")
RESERVE=600             # can_run's margin after the bench: the reduce, the fetch-back and the teardown
MIN_POWER_W=575   # P149's power floor (P222's, kept): a capped board refuses before the install, rc 17
GPU_CLASS=5090; MIN_RAM_GB=32; MIN_DISK_GB=20
SHAKE=0   # p227-shake-N: the shakedown on a cheaper card; never a reading
case "$P227_RUN_ID" in
  p227-5090-1|p227-5090-2|p227-5090-3) :;;
  p227-shake-1|p227-shake-2|p227-shake-3) SHAKE=1;;
  *) say "unregistered run ID: $P227_RUN_ID"; finish 78;;
esac
[ "$SHAKE" = 1 ] && NEED_BENCH=600   # the shakedown's own fit: 4 + 2 layers, 6 reps a cell; well under 600 s
[ -z "${P227_GPU_CLASS:-}${P227_MIN_DISK_GB:-}${P227_MIN_RAM_GB:-}${P227_MIN_POWER_W:-}${P227_NEED_BENCH:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever, engine knob and kernel knob starts unset: the bench sets only E4B_NF4_SMALLM_BW and K25's two
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_INT4_WIDE_TILES E4B_INT4_TILE_PROGRAMS \
      E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN GNF4_GEMV_SPLITK GNF4_GEMV_BW \
      GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_INT8_HEAD_BW E4B_K25_LUT E4B_K25_PLAN E4B_NF4_SMALLM_BW
: > summary.txt; echo "$P227_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W need_bench=$NEED_BENCH" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p227_bench.py p227_reduce.py rr.json.pt test_nf4_smallm_bw_gpu.py test_gemv_bw_served_gpu.py staged-p227.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p227.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p227/staged-p227.sha256"; finish 9; }
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
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P227_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
gpu_snap(){ local q; q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$q" ] || q=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ),$1,${q:-unread}" >> $W/gpu_phases.csv; }
echo "utc,phase,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active" > $W/gpu_phases.csv
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((P227_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P227_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
assert md.version("grouped-nf4-gemm") == "0.50.0", md.version("grouped-nf4-gemm")
import nf4_smallm, int4_b32, nf4_grouped
for mod, want in ((nf4_smallm, os.environ["WANT_SMALLM"]), (int4_b32, os.environ["WANT_B32"]),
                  (nf4_grouped, os.environ["WANT_GROUPED"])):
    got = hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()
    assert got == want, f"{mod.__name__} is {got}, pinned {want}: not the K25, K33 and K39 this lane registers"
assert nf4_smallm._LUT_MODES.get("prmt") == 4 and callable(nf4_grouped._bw_engages), "K39's decode and K33's gate"
from experts4bit_qlora.engines import hot_residency as hr
assert hr._k25_mode_env() == "auto" and hr._lean_glue_mode_env() == "auto", "K25 and K23's lean glue by default"
assert hr.NF4_SMALLM_BW_MAX_T == 8 and callable(hr.resolve_nf4_smallm_bw), "B0 (E4B_NF4_SMALLM_BW) is in the commit"
assert hr._nf4_smallm_bw_env() is False, "the knob is off by default"
# K is each family's served default: qwen3_moe resolves K25's decode to prmt and its plan to c1 (P158, P161); mixtral
# resolves to the tree and today's plan (no read licenses it). The bench sets the same values explicitly.
for mt, lut, plan in (("qwen3_moe", "prmt", "c1"), ("mixtral", "tree", "today")):
    os.environ.pop("E4B_K25_LUT", None); os.environ.pop("E4B_K25_PLAN", None)
    _r = hr.resolve_k25_lut(mt); _p = hr.resolve_k25_plan(_r, model_type=mt)
    assert (_r["lut"], _p["plan"]) == (lut, plan), (mt, _r, _p)
import sys; sys.path.insert(0, "/root/p227"); import p227_reduce as R
assert R.FAMILIES["qwen3"]["k25"] == {"E4B_K25_LUT": "prmt", "E4B_K25_PLAN": "c1"}
assert R.FAMILIES["mixtral"]["k25"] == {"E4B_K25_LUT": "tree", "E4B_K25_PLAN": "today"}
import torch, triton
import experts4bit_qlora as e
open("/root/p227/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p227_bench.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BENCH SELF-TEST FAILED"; finish 21; }
python $W/p227_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "P227 REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card: B0's GPU contract (each row at T = 2..8 bitwise its T == 1 row, K33 only, graph replay;
# both lean-glue settings: 20) and the served T == 1 GEMV route's (5)
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_nf4_smallm_bw_gpu.py test_gemv_bw_served_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^25 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the microbench: per family and T, K / K2 / B in Williams order, a spacer before every timed replay
can_run $NEED_BENCH bench || finish 40
# a shakedown card is not a 5090: K33's auto gate (>= 160 SMs, or sm_89 at 128) would not engage there, so the knob's
# resolver would refuse. A shakedown forces K33 on (the structure, never the speed); a reading keeps every knob unset.
[ "$SHAKE" = 1 ] && export GNF4_GEMV_BW=1
AL=$(step_alarm 2400); say "microbench (alarm=$AL)"; gpu_snap "bench start"
PYTHONPATH= perl -e "alarm $AL; exec @ARGV" python $W/p227_bench.py --replay $W/rr.json.pt --out $W/bench_p227.json $([ "$SHAKE" = 1 ] && echo --shakedown --layers-qwen3 4 --layers-mixtral 2 --reps 6) > logs/bench.log 2>&1
rc=$?; gpu_snap "bench end"
{ echo -n "bench rc=$rc "; grep -a "^P227_BENCH" logs/bench.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
[ "$rc" = 0 ] || { tail -4 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED (rc=$rc) -- the reducer will VOID"; }
say "reduce"; python $W/p227_reduce.py --bench "$W/bench_p227.json" --out "$W/verdict_p227.json" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p227.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p227.json'))['verdict'])")
# a reading finishes 0 only when at least one family was READ (PASS or MISS); a record whose every family is VOID or
# TOLERANCE_FAIL is not a reading (rc 27). A shakedown finishes 0 only on both families' SHAKEDOWN.
READ=$(python -c "import json; v=json.load(open('$W/verdict_p227.json')); print(sum(f.get('verdict') in ('PASS','MISS') for f in (v.get('families') or {}).values()))")
case "$SHAKE/$V" in
  "1/qwen3=SHAKEDOWN mixtral=SHAKEDOWN") finish 0;;
  1/*) say "P227 shakedown $V -- not a clean shakedown"; finish 27;;
  0/*) [ "${READ:-0}" -ge 1 ] && finish 0; say "P227 $V -- no family read"; finish 27;;
esac
