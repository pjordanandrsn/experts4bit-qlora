#!/bin/bash
# bench/p115/p123_run.sh -- lane P123, BOX side (bench/p123/PREREG-p123.md; e4b#1313). Started
# detached by p123_drive.sh with the run's nonce; P123_RUN_NONCE first, then P123_EXIT_CODE.<nonce> + TP_DONE.<nonce>
# on every exit and P123_SUCCESS.<nonce> only when the reducer ran (or, under P123_PROVE=1, when the proof passed).
#
# Where does the shipped default's decode step go, now that the B=1 fused stack (P115, #1361) and the bandwidth
# decode GEMV (P116, grouped-nf4-gemm 0.43.0) are both the default? On ONE RTX 5090:
#   speed    the shipped default server (every lever unset; graphs on, max_seqs 16 named) in two arms of their own
#            processes, S1a S1b, each through P109's W16 and W1: the census's unprofiled reference and its noise, the
#            fusion census and how each knob resolved, grouped-nf4-gemm's dispatch tally, the peak memory;
#   census   B = 1 and 16, each an Nsight Systems graph-mode and node-mode capture of exactly 64 steady decode steps
#            (p123_census.py around SC1b's census_window), exported to sqlite and read by SC1b's sc1b_census.py arm with
#            the v1 NF4 class map (kernel_classes_nf4.json): the step, each kernel class, the in-graph and out-of-graph
#            idle, kernels per step. Skipped when a speed arm failed (the reducer VOIDs);
#   rule     p123_reduce.py: VOID / NOISY / READ; the RESIDUAL row, the GPU busy fraction, the predictions graded.
#   premise  on THIS card, before anything is fetched: the four tests P115 Phase D ran (19 passed, none skipped) (rc 25)
#   order    install + tripwire; self-tests; premise; nsys; fetch; NF4 arena bake; prompts; two speed arms; four
#            captures; the census arms; reduce
#
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), card class 15, disk 13, host RAM
# 16. Only 13 and 18 name the machine for the launcher's next draw. nsys that will not install: 26.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P123_GPU_CLASS P123_MIN_DISK_GB P123_MIN_RAM_GB P123_REHEARSAL P123_SHORT P123_LONG P123_REPS.
# P123_PROVE=1 is the PROVING RUN: everything above, end to end, on ibm-granite/granite-3.1-3b-a800m-instruct at 8 / 24
# tokens and 1 rep (its default is the unfused stack and grouped-nf4-gemm's scalar GEMV; its census reads 32 layers).
set -uo pipefail
W=/root/p123; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p123: $*"; }
NONCE=${P123_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P123_RUN_NONCE.tmp && mv $W/P123_RUN_NONCE.tmp $W/P123_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P123_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P123_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P123_RUN_ID P123_DEADLINE_EPOCH P123_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=6ee2e10408161a9d3c874975c9191a7f2957e6f4   # grouped-nf4-gemm 0.43.0: P116's GEMV is its default; a registered constant
PROVE=${P123_PROVE:-0}
if [ "$PROVE" = 1 ]; then
  MODEL=ibm-granite/granite-3.1-3b-a800m-instruct; REV=a02780686e08a03fe0d2679a293b5c74a90efa89   # P94's pin (SC1's proof model)
  SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; MOE_LAYERS=32
  NEED_FETCH=300; NEED_BAKE=300; NEED_ARM=240; NEED_CAPTURE=420                                  # the proof's own time-left checks
else
  MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                            # SC1's pin
  SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; MOE_LAYERS=48
  NEED_FETCH=1800; NEED_BAKE=1200; NEED_ARM=600; NEED_CAPTURE=900
fi
GPU_CLASS=${P123_GPU_CLASS:-5090}; MIN_DISK_GB=${P123_MIN_DISK_GB:-150}; MIN_RAM_GB=${P123_MIN_RAM_GB:-60}
REHEARSAL=${P123_REHEARSAL:-0}; SHORT=${P123_SHORT:-$SHORT_DEF}; LONG=${P123_LONG:-$LONG_DEF}; REPS=${P123_REPS:-$REPS_DEF}
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
      GNF4_GEMV_SPLITK GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET
: > summary.txt; echo "$P123_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE short=$SHORT long=$LONG reps=$REPS moe_layers=$MOE_LAYERS" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$SHORT" != "$SHORT_DEF" ] || [ "$LONG" != "$LONG_DEF" ] || [ "$REPS" != "$REPS_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p123_box.py p123_census.py p123_reduce.py kernel_classes_nf4.json sc1b_census.py sc1b_e4b_census.py p109_box.py \
         k8_bake.py calib.json test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py \
         test_gemv_bw_served_gpu.py staged-p123.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p123.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p123/staged-p123.sha256"; finish 9; }
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
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint, its NF4 snapshot and arena, four Nsight reports)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P123_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P123_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P123_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
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
assert md.version("grouped-nf4-gemm") == "0.43.0", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged as sp
assert sp._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
# the shipped default (#1361): an unset knob resolves per family -- auto on Qwen3-MoE, 0 on Granite-MoE
unset = {k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}
m, s = sp.resolve_fusion_modes(dict(unset), "qwen3_moe")
assert set(m.values()) == {"auto"} and set(s.values()) == {"default-allowlisted"}, (m, s)
m, s = sp.resolve_fusion_modes(dict(unset), "granitemoe")
assert set(m.values()) == {"0"} and set(s.values()) == {"default-off"}, (m, s)
# P116's route, the default at Qwen3-30B-A3B's two shapes on >= 160 SMs
import nf4_grouped as ng
assert ng._bw() == "auto" and {(1536, 2048), (2048, 768)} <= set(ng._BW_SHAPES), (ng._bw(), ng._BW_SHAPES)
assert {"bw_prmt32", "dotpad", "scalar"} <= set(ng.dispatch_counts()), ng.dispatch_counts()
import fp8_paged_attn, fp8_kv, nvme_arena, int4_b32  # noqa: F401  (decode kernel, KV append, arena bake, the glue folds)
import torch, triton
import experts4bit_qlora as e
open("/root/p123/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p123_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p123_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/sc1b_census.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "CENSUS SELF-TEST FAILED"; finish 21; }
python $W/sc1b_e4b_census.py --selftest | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "WINDOW SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py test_gemv_bw_served_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "19 passed" && ! echo "$LASTL" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as SC1 bakes it) and the prompts
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p109_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the two speed arms: S1a S1b, each a fresh process on the shipped default server (nothing set but the 16 slots)
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
ARMS_OK=1
for TAG in S1a S1b; do
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
  env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
    perl -e "alarm $AL; exec @ARGV" python $W/p123_box.py --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P123_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { ARMS_OK=0; tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
# ---- the census (not when a speed arm failed): Nsight Systems, then B = 1 and 16, graph mode then node mode
install_nsys(){ say "install nsight-systems-cli-$NSYS_VER (NVIDIA devtools apt repo; the image's own nsys is Nsight Compute's, never used)"
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends gnupg2 wget ca-certificates \
    && wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | gpg --dearmor > /usr/share/keyrings/nvidia-devtools-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /" > /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-get update -qq && apt-get install -y -qq --no-install-recommends "nsight-systems-cli-$NSYS_VER"; } > logs/install_nsys.log 2>&1
  NSYS=$(dpkg -L "nsight-systems-cli-$NSYS_VER" 2>/dev/null | grep -E '/nsys$' | head -1)
  if [ -n "$NSYS" ] && [ -x "$NSYS" ]; then
    echo "nsys $("$NSYS" --version 2>&1 | tail -1) at $NSYS" | tee -a versions.txt summary.txt
    "$NSYS" status -e >> logs/install_nsys.log 2>&1; { echo "nsys status -e:"; tail -12 logs/install_nsys.log; } >> forensics.txt
  else say "NSYS INSTALL FAILED: $(tail -1 logs/install_nsys.log | cut -c1-160)"; finish 26; fi; }
nsys_export(){ local rep=$1; [ -s "$rep.nsys-rep" ] || { say "no report $rep.nsys-rep"; return 1; }
  "$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" "$rep.nsys-rep" > "logs/export_$(basename "$rep").log" 2>&1; }
gpu_free(){ local i; for i in $(seq 1 "${1:-120}"); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ] && return 0; sleep 1; done
  say "GPU still held after ${1:-120}s: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; return 1; }
NSYS_VER=2025.6.1; NSYS=""
NSYS_TRACE="-t cuda-sw,nvtx --sample=none --cpuctxsw=none"
SKIP=32; STEPS=64
mkdir -p $W/census
if [ "$ARMS_OK" = 1 ]; then
  install_nsys
  for B in 1 16; do
    for MODE in graph node; do
      S=census_e4b_b${B}_$MODE
      can_run $NEED_CAPTURE "capture $S" || finish 40
      AL=$(step_alarm 1800); say "capture $S (alarm=$AL)"
      # shellcheck disable=SC2086
      env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P123_BATCH=$B P123_OUT=$W/census_drv_b${B}_$MODE.json \
        perl -e "alarm $AL; exec @ARGV" "$NSYS" profile $NSYS_TRACE --cuda-graph-trace=$MODE --capture-range=cudaProfilerApi \
          --capture-range-end=stop --force-overwrite true -o "$W/census/$S" python $W/p123_census.py --prompts $W/prompts.json \
          --skip $SKIP --steps $STEPS > logs/run_$S.log 2>&1
      rc=$?
      pkill -TERM -f "p123_census.py" 2>/dev/null; gpu_free 120 || rc=${rc/#0/32}
      [ "$rc" = 0 ] && { nsys_export "$W/census/$S" || rc=30; }
      { echo -n "capture $S rc=$rc "; grep -a "^P123_CENSUS" logs/run_$S.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt
      [ "$rc" = 0 ] || { tail -4 logs/run_$S.log | cut -c1-300 | tee -a summary.txt; say "CAPTURE $S FAILED (rc=$rc) -- the reducer will VOID"; }
    done
    [ -s $W/census_drv_b${B}_graph.json ] && cp $W/census_drv_b${B}_graph.json $W/census_drv_b$B.json
    W_=W1; [ "$B" = 16 ] && W_=W16
    U=$(python -c "import json,sys; v=[json.load(open(f'$W/arm_{t}.json'))['workloads']['$W_']['decode_ms_per_step'] for t in ('S1a','S1b')]; print(round(sum(v)/len(v),4))" 2>/dev/null)
    G=$W/census/census_e4b_b${B}_graph.sqlite; N=$W/census/census_e4b_b${B}_node.sqlite
    if [ -s "$G" ] && [ -s "$N" ]; then
      python $W/sc1b_census.py arm --graph "$G" --node "$N" --engine e4b --batch $B --classes $W/kernel_classes_nf4.json \
        --delim graph ${U:+--unprofiled-ms $U} --moe-layers $MOE_LAYERS --out $W/census_b$B.json 2>&1 | tail -1 | tee -a summary.txt
    else
      echo "census B=$B UNREAD (missing export) -- the reducer will VOID" | tee -a summary.txt
    fi
  done
else
  echo "a speed arm failed -- the census is not captured (the reducer VOIDs)" | tee -a summary.txt
fi
PF=""; [ "$PROVE" = 1 ] && PF="--proof"
say "reduce"; python $W/p123_reduce.py --dir $W --out $W/verdict_p123.json --e4b-sha $E4B_SHA $PF 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict_p123.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; print(json.load(open('$W/verdict_p123.json'))['verdict'])")
  [ "$V" != VOID ] || { say "PROVE: the reducer VOIDed the proof -- not proved"; finish 27; }
  echo "PROVE -- the proving run: the whole box on $MODEL; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
