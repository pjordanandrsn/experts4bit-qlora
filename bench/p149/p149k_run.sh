#!/bin/bash
# Lane P149, BOX K side (bench/p149/PREREG-p149.md): Mixtral-8x7B's attention projections at decode, cuBLAS bf16 against
# lane P149's prototype W8A16 kernels (an M == 1 GEMV and an M == 16 small-M kernel), in one CUDA-graph harness on one RTX
# 5090. No model. P148's runner (bench/p148/p148_run.sh) by counted edits: its refusals, install and tripwire kept; P136's
# serving premise dropped (no model is served here); the POWER floor added before any install (power.limit >= 575 W, else
# rc 17; an unreadable limit, 15); the kernels' own correctness suite runs COMPILED on the card (TRITON_INTERPRET=0; CI runs
# it interpreted) and must read 135 passed; p149k_bench.py and p149k_reduce.py replace P149's. This script never creates
# compute.
set -uo pipefail
W=/root/p149k; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p149: $*"; }
NONCE=${P149_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P149_RUN_NONCE.tmp && mv $W/P149_RUN_NONCE.tmp $W/P149_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P149_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P149_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P149_RUN_ID P149_DEADLINE_EPOCH P149_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=38b308887688ea5bda7dd449bd3cc3dc1426b995 # v0.47.1 tag commit (e4b CI's pin)
NEED_BENCH=1200   # 2 shapes x 2 M: contract, about 250 selection graphs (each compiles), 4 read graph pairs
RESERVE=600       # every time-left check keeps this much for the fetch and teardown
INSTALL_TO_BENCH=450   # Amendment 3: the install, self-tests and compiled suite before the bench (90 s on p149k-5090-6, x5)
DRY=${P149_DRY:-0}; GPU_CLASS=5090; MIN_RAM_GB=60
case "$P149_RUN_ID" in
  p149k-5090-1|p149k-5090-2|p149k-5090-3|p149k-5090-4|p149k-5090-5|p149k-5090-6|p149k-5090-7|p149k-5090-8|p149k-5090-9) FAMILY=mixtral-attn-int8; MODEL=none; REV=none; MIN_DISK_GB=40;;
  *) say "unregistered run ID: $P149_RUN_ID"; finish 78;;
esac
[ "$DRY" = 0 ] || [ "$FAMILY" = dry ] || { say 'dry mode requires registered A2000 ID'; finish 78; }
[ "$DRY" = 1 ] || [ -z "${P149_GPU_CLASS:-}${P149_MIN_DISK_GB:-}${P149_MIN_RAM_GB:-}${P149_REHEARSAL:-}${P149_MIN_POWER_W:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset (none applies to a no-model microbench; kept from P148)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_KV_STEP_SELECT E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_STEP_TRACE E4B_PAGED_TRACE \
      E4B_PAGED_LAST_LOGITS E4B_FUSE_SWIGLU E4B_FUSE_COMBINE GNF4_PDL GNF4_PDL_MAX_ROWS GNF4_GEMV_DOTPAD GNF4_DECODE_PLAN \
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET
: > summary.txt; echo "$P149_INSTANCE_ID" > INSTANCE_ID; echo "$P149_RUN_ID" > RUN_ID
MIN_POWER_W=575   # P149's power floor (the maintainer's ruling): a capped board refuses before the install, rc 17
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB min_power_w=$MIN_POWER_W family=$FAMILY dry=$DRY" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p149k_bench.py p149_kernels.py p149k_reduce.py test_p149_kernels_interp.py i8_fakequant.py staged-p149k.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p149k.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p149/staged-p149k.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the card class, the disk, the host RAM
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
if [ "$DRY" = 1 ]; then
  case "$GPU_NAME" in *A2000*) ;; *) say "dry mode requires RTX A2000, got $GPU_NAME"; finish 15;; esac
  echo 'DRY_ONLY: staged hashes, CUDA and host inventory checked; no install, fetch, bake, speed arm or trace' | tee -a summary.txt
  : > DRY_ONLY; finish 0
fi
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: 5090 VRAM ${VRAM_MIB:-?} MiB below 30000 MiB floor"; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the installs and their caches)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
# ---- the POWER floor (P149, the maintainer's ruling), before anything is installed: a capped board measures its cap
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
[ -n "$PL" ] || { say "REFUSED: power.limit unreadable"; echo "refused: power.limit unreadable" > REFUSAL; finish 15; }
echo "POWER_LIMIT_W $PL (floor $MIN_POWER_W)" | tee -a summary.txt
[ "$PL" -ge "$MIN_POWER_W" ] || { say "REFUSED: power.limit ${PL} W < ${MIN_POWER_W} W (host-limited power cap)"; echo "refused: power.limit ${PL} W" > REFUSAL; finish 17; }
# Amendment 3: the guard must fit the install, the bench and the reserve, checked BEFORE the install, so a box that cannot
# reach its own read refuses for cents (rc 19, a refusal before the install, not a strike) instead of failing at rc 40.
FIT=$((INSTALL_TO_BENCH + NEED_BENCH + RESERVE)); LEFT=$((P149_DEADLINE_EPOCH - $(date +%s)))
[ "$LEFT" -ge "$FIT" ] || { say "REFUSED: the guard cannot fit the install ${INSTALL_TO_BENCH}s + the bench ${NEED_BENCH}s + the ${RESERVE}s reserve = ${FIT}s; ${LEFT}s left"; echo "refused: guard fit ${LEFT}s < ${FIT}s" > REFUSAL; finish 19; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + RESERVE)) -le "$P149_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s + ${RESERVE}s reserve = $((need + RESERVE))s, only $((P149_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P149_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
assert md.version("grouped-nf4-gemm") == "0.47.1", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
import triton
assert triton.__version__ == "3.4.0", triton.__version__
import sys
sys.path.insert(0, "/root/p149k")
import p149_kernels as K
assert len(K.CONFIGS_GEMV) == 54 and len(K.CONFIGS_SMALL) == 72, "the registered config grids"
import torch, triton
import experts4bit_qlora as e
open("/root/p149k/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p149k_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p149k_bench.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BENCH SELF-TEST FAILED"; finish 21; }
# ---- the kernels' own correctness suite, COMPILED on this card (CI runs it under the interpreter): 135, none skipped
(cd $W && PYTHONPATH='' TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest test_p149_kernels_interp.py -q -rs -p no:cacheprovider) > logs/kernels.log 2>&1
rc=$?; LASTK=$(tail -1 logs/kernels.log)
{ echo -n "kernels rc=$rc: "; echo "$LASTK"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTK" | grep -qE '^135 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "kernels ok" | tee -a summary.txt
else
  echo "kernels failed" | tee -a summary.txt; say "THE PROTOTYPE KERNELS FAIL THEIR OWN SUITE on this card"; finish 25
fi
# ---- the bench: contract (fp64), selection (every passing config), ABBA reads of cuBLAS vs the choice, per shape
can_run $NEED_BENCH bench || finish 40
AL=$(step_alarm 1500); say "bench (alarm=$AL)"; B0=$(date +%s)
env PYTHONPATH= E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA perl -e "alarm $AL; exec @ARGV" python $W/p149k_bench.py --out $W/p149k.json > logs/bench.log 2>&1
rc=$?; echo "BENCH_WALL_S $(( $(date +%s) - B0 ))" | tee -a summary.txt; grep -a "^P149K_" logs/bench.log | cut -c1-200 | tee -a summary.txt
[ "$rc" = 0 ] && [ -s $W/p149k.json ] || { tail -6 logs/bench.log | cut -c1-300 | tee -a summary.txt; say "BENCH FAILED rc=$rc"; finish 34; }
say "reduce"; python $W/p149k_reduce.py --bench $W/p149k.json --out $W/verdict_p149k.json --e4b-sha "$E4B_SHA" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p149k.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p149k.json'))['verdict'])")
case "$V" in K_GO|KERNEL_NO_GAIN) ;; *) say "P149 box K $V -- no reading (NOISY and VOID are not readings)"; finish 27;; esac
finish 0
