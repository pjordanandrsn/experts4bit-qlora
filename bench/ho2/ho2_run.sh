#!/bin/bash
# HO2's box runner (bench/ho2/PREREG-ho2.md): the planner's own plan, executed through Loggetta's CLI, for two MoE families
# that no planner constant has seen, at two sequence lengths, on this box's card. P224's runner (bench/p224/p224_run.sh)
# for the refusals, the guard fit and the install; the arms are this lane's four `loggetta train` runs and the reduce.
# Phase stamps by bench/common/phases.sh. No instance id and no host name are written here. This script never creates
# compute.
set -uo pipefail
W=/root/ho2; mkdir -p $W/logs $W/run; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] ho2: $*"; }
NONCE=${HO2_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/HO2_RUN_NONCE.tmp && mv $W/HO2_RUN_NONCE.tmp $W/HO2_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > HO2_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > HO2_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in HO2_RUN_ID HO2_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7       # e4b CI's grouped-nf4-gemm pin (.github/workflows/ci.yml)
LOGGETTA_SHA=LOGGETTA_0_6_2_RELEASE_SHA                  # loggetta 0.6.2's release commit, filled in at registration
ALPACA_REV=dce01c9b08f87459cf36a430d809084718273017       # tatsu-lab/alpaca's pinned text column (P224 Amendment 1's pin)
# The two families, named before any measurement (PREREG, "Families")
LFM2=LiquidAI/LFM2-8B-A1B; LFM2_REV=c1c44ff9fc00db3ebf4516970563f5f383d23670
ERNIE=baidu/ERNIE-4.5-21B-A3B-PT; ERNIE_REV=87db95487941cb39592ee0abca3b9155a6d19c5c
case "$HO2_RUN_ID" in
  ho2-5090-1|ho2-5090-2|ho2-5090-3) GPU_CLASS="RTX 5090"; CC_WANT=12.0; VRAM_FLOOR=30000;;
  ho2-4090-1|ho2-4090-2|ho2-4090-3) GPU_CLASS="RTX 4090"; CC_WANT=8.9; VRAM_FLOOR=23000;;
  *) say "unregistered run ID: $HO2_RUN_ID"; finish 78;;
esac
INSTALL_TO_GATE_S=450; NEED_FETCH=900; NEED_TRAIN=480; NEED_REDUCE=60; RESERVE=600
NEED_WORK=$((NEED_FETCH + 4 * NEED_TRAIN + NEED_REDUCE))
MIN_RAM_GB=48; MIN_DISK_GB=120
[ -z "${HO2_GPU_CLASS:-}${HO2_MIN_DISK_GB:-}${HO2_NEED_TRAIN:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every estimate-read switch starts unset: the planner prices e4b's defaults and the run runs them (recipe.estimate_env)
unset E4B_ABSMAX_DQ E4B_TRAIN_FUSE_QKV E4B_CHUNKED_LM_LOSS E4B_CHUNKED_EVAL_LOSS E4B_CKPT_OFFLOAD OFFLOAD_EXPERTS TRAIN_ARENA \
      NF4_QLORA_PAD_BUCKETS NF4_QLORA_SINGLE_LADDER E4B_COMBINE_CHUNK TRITON_INTERPRET
: > summary.txt; echo "$HO2_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA loggetta=$LOGGETTA_SHA alpaca=$ALPACA_REV class=$GPU_CLASS need_work=$NEED_WORK" | tee -a summary.txt
case "$LOGGETTA_SHA" in *[!0-9a-f]*|"") say "refusing: LOGGETTA_SHA is not a pinned commit ($LOGGETTA_SHA)"; finish 78;; esac
# ---- staged pieces, byte-for-byte
for f in ho2_reduce.py phases.sh staged-ho2.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-ho2.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/ho2/staged-ho2.sha256"; finish 9; }
# ---- refusals before anything is installed: CUDA, the card, the disk, the host RAM, the guard fit
CUDA_PROBE=$(python -c "import torch; print('ok' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'no-cuda ' + torch.__version__)" 2>/dev/null | tail -1)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in ok) ;; no-cuda*) echo "refused: cuda unusable" > REFUSAL; finish 18;; *) say "DUD BOX"; finish 10;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit --format=csv,noheader | tee forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the run ID registers $GPU_CLASS"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = "$CC_WANT" ] || { say "REFUSED: compute capability must be $CC_WANT, got $CC"; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge "$VRAM_FLOOR" ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB below ${VRAM_FLOOR} MiB"; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; finish 16; }
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_WORK + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((HO2_DEADLINE_EPOCH - $(date +%s)))
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
source $W/phases.sh || finish 9
export PHASES_FILE=$W/run/phases.jsonl
can_run(){ local need=$1 left; left=$((HO2_DEADLINE_EPOCH - $(date +%s))); [ $((need + RESERVE)) -le "$left" ] || { say "STOP-2: $2 needs $((need + RESERVE))s, ${left}s left -- skipped"; echo "SKIPPED $2" >> summary.txt; return 1; }; }
alarm(){ local cap=$1 left=$(( HO2_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
# ---- the install: torch held at the image's, then e4b (with `datasets`, which Loggetta's data path imports), gnf4, Loggetta
phase_start install setup
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
echo "torch==2.8.0" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" "datasets>=2.14" accelerate sentencepiece safetensors "huggingface_hub>=0.23" numpy \
  || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { say "PIP FAIL (gnf4)"; finish 9; }
pipx logs/pip_logg.log 900 --no-deps "git+https://github.com/pjordanandrsn/loggetta.git@$LOGGETTA_SHA" || { say "PIP FAIL (loggetta)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_LOGG=$LOGGETTA_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
for dist, want in (("experts4bit-qlora", "WANT_E4B"), ("grouped-nf4-gemm", "WANT_GNF4"), ("loggetta", "WANT_LOGG")):
    d = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
    assert d.get("vcs_info", {}).get("commit_id") == os.environ[want], f"installed {dist} is not the pinned commit: {d}"
import torch, datasets  # noqa: F401  -- Loggetta's data path imports datasets for every dataset, local or Hub
import loggetta.data, loggetta.backends.experts4bit_train  # noqa: F401,E401  -- a missing module refuses here, before any fetch
assert torch.__version__ == "2.8.0+cu128", torch.__version__
open("/root/ho2/versions.txt", "w").write("".join(f"{n} {md.version(n)}\n" for n in
     ("experts4bit-qlora", "grouped-nf4-gemm", "loggetta", "torch", "transformers", "bitsandbytes", "datasets")) + f"cc {torch.cuda.get_device_capability()}\n")
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
python $W/ho2_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
phase_end install
# ---- the two checkpoints, at their pins
can_run $NEED_FETCH fetch || finish 40
phase_start fetch setup
for spec in "$LFM2 $LFM2_REV" "$ERNIE $ERNIE_REV"; do
  set -- $spec
  SNAP=$(perl -e "alarm $(alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$2', max_workers=8))" 2>> logs/fetch.log | tail -1)
  [ -d "$SNAP" ] || { tail -2 logs/fetch.log; say "DL FAIL $1"; finish 11; }
  echo "FETCHED $1@$2" | tee -a summary.txt
done
phase_end fetch
# ---- the four rows: the planner's own plan (no receipts on file, no --fix), executed by `loggetta train`
train(){ local tag=$1 model=$2 rev=$3 seq=$4
  can_run $NEED_TRAIN "train $tag" || return 1
  phase_start "train_$tag" measure
  perl -e "alarm $(alarm 1800); exec @ARGV" python -m loggetta train "$model" --revision "$rev" \
    --dataset tatsu-lab/alpaca --dataset-revision $ALPACA_REV --format text --text-field text \
    --seq "$seq" --micro-batch 2 --steps 20 --seed 0 \
    --out $W/run/receipts_$tag --adapter-out $W/run/adapter_$tag > logs/train_$tag.log 2>&1
  local rc=$?; phase_end "train_$tag"
  echo "train $tag rc=$rc" | tee -a summary.txt
  return $rc; }
train lfm2_1024 $LFM2 $LFM2_REV 1024 || say "train lfm2_1024 failed (rc $?) -- the reducer will VOID that row"
train lfm2_2048 $LFM2 $LFM2_REV 2048 || say "train lfm2_2048 failed (rc $?) -- the reducer will VOID that row"
train ernie_1024 $ERNIE $ERNIE_REV 1024 || say "train ernie_1024 failed (rc $?) -- the reducer will VOID that row"
train ernie_2048 $ERNIE $ERNIE_REV 2048 || say "train ernie_2048 failed (rc $?) -- the reducer will VOID that row"
# ---- the reduce
phase_start reduce reduce
python $W/ho2_reduce.py --dir $W/run --run-id "$HO2_RUN_ID" --e4b "$E4B_SHA" --gnf4 "$GNF4_SHA" --loggetta "$LOGGETTA_SHA" \
  --out $W/run/verdict_ho2.json 2>&1 | tee -a summary.txt
phase_end reduce
[ -s $W/run/verdict_ho2.json ] || { say "REDUCER FAILED"; finish 22; }
python -c "import json; print('HO2', json.load(open('$W/run/verdict_ho2.json'))['verdict'])" | tee -a summary.txt
finish 0
