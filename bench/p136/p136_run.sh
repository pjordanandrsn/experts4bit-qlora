#!/bin/bash
# P136 R-e box runner. Derived from P123's served Nsight census by named
# substitutions: run ID/family pins, W16-only gpt-oss, and a separate reducer.
# Registration: PREREG-p136.md. This script never creates compute.
set -uo pipefail
W=/root/p136; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p136: $*"; }
NONCE=${P136_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P136_RUN_NONCE.tmp && mv $W/P136_RUN_NONCE.tmp $W/P136_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P136_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P136_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P136_RUN_ID P136_DEADLINE_EPOCH P136_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=70bc03cb7a61b03c4e801e7423c58c103a1b6d7d # v0.46.1 tag commit
SHORT=32; LONG=160; REPS=3; NEED_FETCH=3600; NEED_BAKE=1800; NEED_ARM=1800; NEED_CAPTURE=1800
DRY=${P136_DRY:-0}; GPU_CLASS=5090; MIN_RAM_GB=60
case "$P136_RUN_ID" in
  p136-gemma4-5090-1|p136-gemma4-5090-2) FAMILY=gemma4; MODEL=google/gemma-4-26B-A4B-it; REV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52; MOE_LAYERS=30; MIN_DISK_GB=200; BATCHES='1 16';;
  p136-mixtral-5090-1|p136-mixtral-5090-2) FAMILY=mixtral; MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; REV=eba92302a2861cdc0098cc54bc9f17cb2c47eb61; MOE_LAYERS=32; MIN_DISK_GB=220; BATCHES='1 16';;
  p136-gptoss-5090-1|p136-gptoss-5090-2) FAMILY=gptoss; MODEL=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee; MOE_LAYERS=24; MIN_DISK_GB=200; BATCHES='16';;
  p136-a2000-dry-1) [ "$DRY" = 1 ] || { say 'dry id requires P136_DRY=1'; finish 78; }; FAMILY=dry; MODEL=none; REV=none; MOE_LAYERS=0; MIN_DISK_GB=0; BATCHES='';;
  *) say "unregistered run ID: $P136_RUN_ID"; finish 78;;
esac
[ "$DRY" = 0 ] || [ "$FAMILY" = dry ] || { say 'dry mode requires registered A2000 ID'; finish 78; }
[ "$DRY" = 1 ] || [ -z "${P136_GPU_CLASS:-}${P136_MIN_DISK_GB:-}${P136_MIN_RAM_GB:-}${P136_REHEARSAL:-}${P136_SHORT:-}${P136_LONG:-}${P136_REPS:-}" ] || { say 'reading knobs are frozen'; finish 78; }
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
: > summary.txt; echo "$P136_INSTANCE_ID" > INSTANCE_ID; echo "$P136_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB family=$FAMILY dry=$DRY short=$SHORT long=$LONG reps=$REPS moe_layers=$MOE_LAYERS" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p136_box.py p136_census.py p136_reduce.py kernel_classes_nf4.json sc1b_census.py sc1b_e4b_census.py p109_box.py \
         k8_bake.py calib.json test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py \
         test_gemv_bw_served_gpu.py staged-p136.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p136.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p136/staged-p136.sha256"; finish 9; }
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
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (family checkpoint, NF4 snapshot, arena and Nsight reports)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P136_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P136_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P136_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
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
assert md.version("grouped-nf4-gemm") == "0.46.1", md.version("grouped-nf4-gemm")
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged as sp
assert sp._graphs_env("", "cuda", "all-vram") is True, "the subject is the graph default"
import nf4_grouped as ng
assert "scalar" in ng.dispatch_counts(), ng.dispatch_counts()
import fp8_paged_attn, fp8_kv, nvme_arena, int4_b32  # noqa: F401  (decode kernel, KV append, arena bake, the glue folds)
import torch, triton
import experts4bit_qlora as e
open("/root/p136/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\nsm_count {torch.cuda.get_device_properties(0).multi_processor_count}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p136_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p136_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
python $W/sc1b_census.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "CENSUS SELF-TEST FAILED"; finish 21; }
python $W/sc1b_e4b_census.py --selftest | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "WINDOW SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py test_kv_step_select.py test_fused_glue_decode_graphs_gpu.py test_gemv_bw_served_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
{ echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -qE '^19 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))?$'; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# P123's recorded build, not merely its major/minor label. A missing or
# different profiler refuses before checkpoint download.
NSYS_VER=2025.6.1
NSYS_EXPECT='NVIDIA Nsight Systems version 2025.6.1.190-256136895201v0'
NSYS=""
nsys_pick(){
  local candidate version
  for candidate in "$(command -v nsys 2>/dev/null || true)" \
      /opt/nvidia/nsight-systems-cli/2025.6.1/target-linux-x64/nsys \
      $(dpkg -L "nsight-systems-cli-$NSYS_VER" 2>/dev/null | grep -E '/nsys$' || true); do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    version=$("$candidate" --version 2>&1 | tail -1)
    if [ "$version" = "$NSYS_EXPECT" ]; then NSYS=$candidate; return 0; fi
  done
  return 1
}
install_nsys(){
  if ! nsys_pick; then
    say "exact P123 nsys absent; install nsight-systems-cli-$NSYS_VER before fetch"
    # The devtools repo is currently signed by the 7fa2af80 key; NVIDIA's
    # CUDA repo rotated to 3bf863cc. Accept only these two verified key
    # fingerprints, scoped to this devtools apt source, then pin the CLI build.
    if ! { apt-get update -qq && apt-get install -y -qq --no-install-recommends gnupg2 wget ca-certificates \
      && wget -qO "$W/nvidia-devtools-old.pub" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
      && wget -qO "$W/nvidia-devtools-new.pub" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
      && FPR_OLD=$(gpg --show-keys --with-colons "$W/nvidia-devtools-old.pub" 2>/dev/null | awk -F: '$1=="fpr" {print $10; exit}') \
      && FPR_NEW=$(gpg --show-keys --with-colons "$W/nvidia-devtools-new.pub" 2>/dev/null | awk -F: '$1=="fpr" {print $10; exit}') \
      && [ "$FPR_OLD" = AE09FE4BBD223A84B2CCFCE3F60F4B3D7FA2AF80 ] \
      && [ "$FPR_NEW" = EB693B3035CD5710E231E123A4B469963BF863CC ] \
      && gpg --batch --yes --dearmor -o "$W/nvidia-devtools-old.gpg" "$W/nvidia-devtools-old.pub" \
      && gpg --batch --yes --dearmor -o "$W/nvidia-devtools-new.gpg" "$W/nvidia-devtools-new.pub" \
      && cat "$W/nvidia-devtools-old.gpg" "$W/nvidia-devtools-new.gpg" > /usr/share/keyrings/nvidia-devtools-keyring.gpg \
      && . /etc/os-release && [ "$ID" = ubuntu ] && [ "$(dpkg --print-architecture)" = amd64 ] \
      && DISTRO=ubuntu$(echo "$VERSION_ID" | tr -d .) \
      && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/$DISTRO/amd64/ /" > /etc/apt/sources.list.d/nvidia-devtools.list \
      && apt-get update -qq && apt-get install -y -qq --no-install-recommends "nsight-systems-cli-$NSYS_VER"; } > logs/install_nsys.log 2>&1; then
      say "NSYS INSTALL REFUSED before fetch: $(tail -1 logs/install_nsys.log | cut -c1-160)"
      echo 'refused: exact nsys unavailable' > REFUSAL; finish 26
    fi
    nsys_pick || { say "NSYS VERSION REFUSED before fetch: need $NSYS_EXPECT"; echo 'refused: nsys version mismatch' > REFUSAL; finish 26; }
  fi
  echo "nsys $NSYS_EXPECT at $NSYS" | tee -a versions.txt summary.txt
  # P123's successful 5090 capture reported CPU profiling failures here;
  # our --sample=none CUDA trace does not require those optional features.
  "$NSYS" status -e > logs/nsys_status.log 2>&1
  echo "nsys status rc=$? (diagnostic only; CUDA trace is checked at capture)" >> forensics.txt
  { echo 'nsys status -e:'; tail -12 logs/nsys_status.log; } >> forensics.txt
}
install_nsys
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
perl -e "alarm 1200; exec @ARGV" python $W/p109_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P109_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the two speed arms: S1a S1b, each a fresh process on the shipped default server (nothing set but the 16 slots)
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16"
[ "$FAMILY" = mixtral ] && ENGINE_ENV="$ENGINE_ENV E4B_PAGED_MAX_TOKENS_PER_SEQ=768"
ARMS_OK=1
for TAG in S1a S1b; do
  can_run $NEED_ARM "arm $TAG" || finish 40
  AL=$(step_alarm 2400); say "arm $TAG (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
  env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P136_FAMILY=$FAMILY \
    perl -e "alarm $AL; exec @ARGV" python $W/p136_box.py --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P136_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { ARMS_OK=0; tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
# ---- the census (not when a speed arm failed): Nsight Systems, then B = 1 and 16, graph mode then node mode
nsys_export(){ local rep=$1; [ -s "$rep.nsys-rep" ] || { say "no report $rep.nsys-rep"; return 1; }
  "$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" "$rep.nsys-rep" > "logs/export_$(basename "$rep").log" 2>&1; }
gpu_free(){ local i; for i in $(seq 1 "${1:-120}"); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ] && return 0; sleep 1; done
  say "GPU still held after ${1:-120}s: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; return 1; }
NSYS_TRACE="-t cuda-sw,nvtx --sample=none --cpuctxsw=none"
SKIP=32; STEPS=64
mkdir -p $W/census
if [ "$ARMS_OK" = 1 ]; then
  for B in $BATCHES; do
    for MODE in graph node; do
      S=census_e4b_b${B}_$MODE
      can_run $NEED_CAPTURE "capture $S" || finish 40
      AL=$(step_alarm 1800); say "capture $S (alarm=$AL)"
      # shellcheck disable=SC2086
      env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P136_BATCH=$B P136_OUT=$W/census_drv_b${B}_$MODE.json \
        perl -e "alarm $AL; exec @ARGV" "$NSYS" profile $NSYS_TRACE --cuda-graph-trace=$MODE --capture-range=cudaProfilerApi \
          --capture-range-end=stop --force-overwrite true -o "$W/census/$S" python $W/p136_census.py --prompts $W/prompts.json \
          --skip $SKIP --steps $STEPS > logs/run_$S.log 2>&1
      rc=$?
      gpu_free 120 || rc=32
      [ "$rc" = 0 ] && { nsys_export "$W/census/$S" || rc=30; }
      { echo -n "capture $S rc=$rc "; grep -a "^P136_CENSUS" logs/run_$S.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt
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
say "reduce"; python $W/p136_reduce.py --dir "$W" --out "$W/verdict_p136.json" --e4b-sha "$E4B_SHA" --family "$FAMILY" 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p136.json" ] || { say "REDUCER FAILED"; finish 22; }
V=$(python -c "import json; print(json.load(open('$W/verdict_p136.json'))['verdict'])")
[ "$V" = READ ] || { say "census $V -- no family reading"; finish 27; }
finish 0
