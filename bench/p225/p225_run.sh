#!/bin/bash
# P225's box runner (bench/p225/PREREG-p225.md): P224's runner (bench/p224/p224_run.sh) by named edits, P224 CLOSED.
# The edits: P225's run ids (p225-5090-N, at the 30-Mops floor; no fallback tier at the registered N);
# the on-box host screen BEFORE any install (p225_probe.py; a refusal is HOST_REFUSED, rc 19, never a strike); the
# strike-eligibility marker at the first install; every phase alarm a named constant (tests/test_p225.py parses
# them); a Mops re-probe at every phase boundary; separate REF and HF0 stamps inside the reference process; P225's
# canary (p225_canary.py); and, after P224's collect, P225's adapter gate (p225_gate.py) and verdict (p225_reduce.py).
# No shakedown and no fixture generation. No instance id and no host name are written here. This script never creates
# compute.
set -uo pipefail
W=/root/p225; mkdir -p $W/logs $W/run; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p225: $*"; }
NONCE=${P225_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P225_RUN_NONCE.tmp && mv $W/P225_RUN_NONCE.tmp $W/P225_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P225_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P225_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P225_RUN_ID P225_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7       # e4b CI's grouped-nf4-gemm pin (.github/workflows/ci.yml)
LOGGETTA_SHA=90a7697d0e6d6cdfa897cb5d629699b56f37cfe9  # loggetta main after #84: 0.6.1 + the torch preflight (not 0.6.2)
CANARY_SHA256=91c24ea851fec34c4e74cb589a0e42cbed2e0b84f1af813f013a57f8cda7da07   # P225's held-out set (p225_canary.py, K topics by the sizing rule)
INSTALL_TO_GATE_S=450   # the install and the self-tests (P224's)
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39   # the SC1 pin, as P224
case "$P225_RUN_ID" in
  p225-5090-1|p225-5090-2|p225-5090-3) TIER=a;;        # the 30-Mops floor (the ruling); no fallback tier at this N
  *) say "unregistered run ID: $P225_RUN_ID"; finish 78;;
esac
# ---- every fixed need and alarm (PREREG "The alarms"; p225_derive.py writes them, tests/test_p225.py parses them)
NEED_FETCH=2120; NEED_BAKE=550; NEED_TRAIN=1840; NEED_REF=8600; NEED_SLOT=150; NEED_SCORE=550
FETCH_ALARM=2650; BAKE_ALARM=690; TRAIN_ALARM=2300; REF_ALARM=10750; SLOT_ALARM=880
GPU_CLASS=5090; VRAM_FLOOR=30000
NEED_GATE=60; NEED_REDUCE=60; RESERVE=600
SHAKE=0   # P225 registers no shakedown; P224's shakedown branches below never run
NEED_WORK=$((NEED_FETCH + NEED_BAKE + NEED_TRAIN + NEED_REF + 6 * NEED_SLOT + 2 * NEED_SCORE + NEED_GATE + NEED_REDUCE))
# the reading fetches Qwen3-30B-A3B (about 61 GB), bakes the arena (about 16 GB) and installs (about 5 GB): a box
# with less than 100 GB free refuses here, before any install, instead of failing mid-fetch (the manifest rents 120)
MIN_RAM_GB=16; MIN_DISK_GB=90   # P224's checks kept below the screen's floors (60 GiB RAM; 100 GB, where df -BG
# counts GiB), so a floor failure is always the screen's HOST_REFUSED (rc 19), never an inherited refusal code
[ -z "${P225_GPU_CLASS:-}${P225_MIN_DISK_GB:-}${P225_NEED_SLOT:-}${P225_TIER:-}" ] || { say 'reading knobs are frozen'; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset: the server's own defaults; the adapter knobs are set per arm below, never here
unset E4B_PAGED_ADAPTER E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE E4B_PAGED_ADAPTER_ALLOW_UNVERIFIED_ARENA E4B_ABSMAX_DQ \
      E4B_SERVE_EXP_INT4 E4B_SERVE_ATTN_INT4 E4B_SERVE_ATTN_INT8 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_FUSE_QKV E4B_PAGED_BULK_KV E4B_PAGED_PREFILL_GRAPH \
      E4B_PAGED_DECODE_LOOKAHEAD E4B_PAGED_LAST_LOGITS E4B_ATTN_SPLIT_CAP E4B_INT8_HEAD_BW E4B_K25_LUT E4B_K25_PLAN \
      E4B_NF4_GROUPED_SMALLM E4B_DENSE_GEMV_BW GNF4_GEMV_BW GNF4_GEMV_BW_PLAN GNF4_GEMV_BW_DECODE TRITON_INTERPRET E4B_SPEC
: > summary.txt; echo "$P225_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA loggetta=$LOGGETTA_SHA model=$MODEL@$REV tier=$TIER need_work=$NEED_WORK" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in p225_probe.py p225_gate.py p225_reduce.py p225_canary.py p224_box.py p224_canary.py p224_reduce.py phases.sh calib.json test_adapter_schema.py staged-p225.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p225.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p225/staged-p225.sha256"; finish 9; }
# ---- refusals before anything is installed: CUDA, the card, the disk, the host RAM, the guard fit
CUDA_PROBE=$(python -c "import torch; print('ok' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'no-cuda ' + torch.__version__)" 2>/dev/null | tail -1)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in ok) ;; no-cuda*) echo "refused: cuda unusable" > REFUSAL; finish 18;; *) say "DUD BOX"; finish 10;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap,power.limit --format=csv,noheader | tee forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "$SHAKE" = 1 ]; then
  case "$CC" in 8.6|8.9|9.0|10.0|12.0) ;; *) say "REFUSED: a shakedown needs compute capability 8.6 or later, got $CC"; finish 15;; esac
else
  case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME'"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
  [ "$CC" = 12.0 ] || { say "REFUSED: 5090 compute capability must be 12.0, got $CC"; finish 15; }
fi
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge "$VRAM_FLOOR" ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB below ${VRAM_FLOOR} MiB"; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; finish 16; }
# ---- P225's host screen, ON THE BOX, BEFORE any install (PREREG "The host screen"): the registered floors only, refused
# mechanically as HOST_REFUSED (rc 19, never a strike); a screen that cannot run refuses the same way
python $W/p225_probe.py --floors $TIER --work $W --model $MODEL --revision $REV > logs/screen.log 2>&1; SCR=$?
grep -a "^P225_SCREEN" logs/screen.log | tee -a summary.txt
[ "$SCR" = 0 ] || { say "HOST_REFUSED: the screen read rc $SCR (a floor failed or the screen could not run); never a strike"; echo "refused: host screen rc $SCR" > REFUSAL; finish 19; }
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_WORK + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((P225_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
source $W/phases.sh || finish 9          # the phase stamps, after the fit check (bench/common/README.md, "Guard fit")
export PHASES_FILE=$W/run/phases.jsonl
can_run(){ local need=$1 left; left=$((P225_DEADLINE_EPOCH - $(date +%s))); [ $((need + RESERVE)) -le "$left" ] || { say "STOP-2: $2 needs $((need + RESERVE))s, ${left}s left -- skipped"; echo "SKIPPED $2" >> summary.txt; return 1; }; }
alarm(){ local cap=$1 left=$(( P225_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
# a Mops re-probe and the steal share at every phase boundary: recorded, never refusing (PREREG "The host screen")
boundary(){ python $W/p225_probe.py --boundary "$1" 2>/dev/null | tee -a summary.txt; }
# ---- the install: torch held at the image's (never a pip default torch the driver cannot run), then e4b, gnf4, Loggetta
date -u +%FT%TZ > $W/run/STRIKE_ELIGIBLE   # every floor passed: from the first install on, a failure can be a strike
phase_start install setup
TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_FULL" = 2.8.0+cu128 ] || { say "REFUSED: torch image $TORCH_FULL is not 2.8.0+cu128"; finish 28; }
echo "torch==2.8.0" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy \
  fastapi "uvicorn==0.38.0" httpx "datasets>=2.14" || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
# Loggetta goes in --no-deps below (its own requirements would re-resolve e4b[train,fast] and grouped-nf4-gemm from
# PyPI over the pinned commits), so the one it needs on the experts4bit path, datasets, is named above under the held
# torch (peft is imported only by its dense backend, which P224 never runs)
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { say "PIP FAIL (gnf4)"; finish 9; }
pipx logs/pip_logg.log 900 --no-deps "git+https://github.com/pjordanandrsn/loggetta.git@$LOGGETTA_SHA" || { say "PIP FAIL (loggetta)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_LOGG=$LOGGETTA_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
for dist, want in (("experts4bit-qlora", "WANT_E4B"), ("grouped-nf4-gemm", "WANT_GNF4"), ("loggetta", "WANT_LOGG")):
    d = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
    assert d.get("vcs_info", {}).get("commit_id") == os.environ[want], f"installed {dist} is not the pinned commit: {d}"
import torch
assert torch.__version__ == "2.8.0+cu128", torch.__version__          # the held torch survived every install
from experts4bit_qlora.adapter_schema import quant_fingerprint          # the schema v2 code G5 runs
import datasets                                                         # Loggetta's data path (loggetta/data.py imports it lazily)
import nvme_bake_nf4                                                    # Amendment 3: the arena's bake (grouped-nf4-gemm)
print("datasets", datasets.__version__)
import loggetta.data, loggetta.backends.experts4bit_train, loggetta.backends.experts4bit_adapters   # the train path
from experts4bit_qlora import adapter_dir
assert hasattr(adapter_dir, "merge_attention_lora")
open("/root/p225/versions.txt", "w").write("".join(f"{n} {md.version(n)}\n" for n in
     ("experts4bit-qlora", "grouped-nf4-gemm", "loggetta", "torch", "transformers")) + f"cc {torch.cuda.get_device_capability()}\n")
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
for t in p224_box p224_canary p224_reduce p225_probe p225_gate p225_reduce p225_canary; do
  python $W/$t.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "$t SELF-TEST FAILED"; finish 21; }; done
phase_end install
boundary install
# ---- gate G5's registered CUDA offload test, on this card
phase_start offload_test measure
(cd $W && PYTHONPATH='' perl -e 'alarm 600; exec @ARGV' python -m pytest test_adapter_schema.py -q -rs -p no:cacheprovider \
  -k test_real_offload_fingerprints_as_resident_on_cuda) > run/offload_test.log 2>&1
echo "offload test: $(tail -1 run/offload_test.log)" | tee -a summary.txt
phase_end offload_test
# ---- the canary data, its pin recomputed here
python $W/p225_canary.py --out $W/run/data | tee -a summary.txt
grep -q "^CANARY_SHA256 $CANARY_SHA256\$" summary.txt || { say "CANARY DATA IS NOT THE REGISTERED PIN"; finish 9; }
# ---- the checkpoint and its NF4 arena: bake_nf4 from the HF snapshot (Amendment 3), its provenance checked before any server
can_run $((NEED_FETCH + NEED_BAKE)) fetch || finish 40
phase_start fetch setup
SNAP=$(perl -e "alarm $(alarm $FETCH_ALARM); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', max_workers=8))" 2> logs/fetch.log | tail -1)
[ -d "$SNAP" ] || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
phase_end fetch
boundary fetch
phase_start bake setup
mkdir -p $W/work
perl -e "alarm $(alarm $BAKE_ALARM); exec @ARGV" python -c "from nvme_bake_nf4 import bake_nf4; bake_nf4('$SNAP', '$W/work/nf4.arena')" \
  > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
phase_end bake
boundary bake
# the arena's provenance, before any server process: the paged server's refusals 3 and 4 read exactly these fields
python - "$W/work/nf4.arena.index.json" "$MODEL" "$REV" > logs/arena_check.log 2>&1 <<'PY'
import json, sys
from experts4bit_qlora.adapter_dir import arena_source
idx = json.load(open(sys.argv[1]))
got, q = arena_source(idx), idx.get("quantizer") or {}
print(f"ARENA source={got[0]}@{got[1]} quantizer={q.get('quant_type')}/{q.get('blocksize')}")
sys.exit(0 if got == (sys.argv[2], sys.argv[3]) and (q.get("quant_type"), q.get("blocksize")) == ("nf4", 64) else 1)
PY
ARC=$?; cat logs/arena_check.log | tee -a summary.txt
[ "$ARC" = 0 ] || { say "ARENA PROVENANCE is not ($MODEL, $REV, nf4/64) -- HARNESS_ERROR before any slot"; finish 12; }
python - "$P225_RUN_ID" "$SHAKE" "$E4B_SHA" "$MODEL" "$REV" > $W/run/meta_in.json <<'PY'
import json, sys, torch
rid, shake, e4b, model, rev = sys.argv[1:]
print(json.dumps({"run_id": rid, "shakedown": shake == "1", "e4b_sha": e4b, "model": model, "model_revision": rev,
                  "device": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0))}))
PY
# ---- the canary train (Loggetta's CLI; Amendment 3: canary A only, its kernel reference by Loggetta's plan), then its record
train(){ local arm=$1 dq=$2
  can_run $NEED_TRAIN "train $arm" || return 1
  phase_start "train_$arm" measure
  env ${dq:+E4B_ABSMAX_DQ=$dq} perl -e "alarm $(alarm $TRAIN_ALARM); exec @ARGV" python -m loggetta train "$MODEL" --revision "$REV" \
    --dataset $W/run/data/train.jsonl --format chat --loss assistant --packing concat --repeat-data --seq 512 --micro-batch 1 \
    --grad-accum 1 --steps 80 --learning-rate 5e-4 --lr-schedule cosine --seed 0 --fix train_experts=false \
    --fix train_attention=true --fix attn_4bit=false --fix expert_residency=device \
    --out $W/run/receipts_$arm --adapter-out $W/run/adapter_$arm > logs/train_$arm.log 2>&1
  local rc=$?; phase_end "train_$arm"; boundary "train_$arm"
  python $W/p224_box.py --mode train-record --arm $arm --receipts $W/run/receipts_$arm --adapter $W/run/adapter_$arm \
    --out $W/run/train_$arm.json | tee -a summary.txt
  return $rc; }
train REF 0 || say "train REF failed (rc $?) -- the reducer will VOID"
# ---- the reference arm: REF (and HF0, its zeroed twin, and the fingerprint seconds)
can_run $NEED_REF reference && {
  phase_start ref_REF measure
  # each output line stamped with its epoch second, so the receipt splits REF from HF0 inside the one process
  E4B_ABSMAX_DQ=0 perl -e "alarm $(alarm $REF_ALARM); exec @ARGV" python $W/p224_box.py --mode ref --arm REF --adapter $W/run/adapter_REF \
    --heldout $W/run/data/heldout.jsonl --out $W/run/REF.json --hf0-out $W/run/HF0.json \
    --fingerprint-out $W/run/fingerprint_REF.json 2>&1 | while IFS= read -r l; do printf '[%s] %s\n' "$(date +%s)" "$l"; done > logs/ref_REF.log
  phase_end ref_REF; boundary ref_REF
  grep -a "P224_REF REF \|P224_REF HF0 " logs/ref_REF.log | sed 's/^\[\([0-9]*\)\] /P225_REF_STAMP \1 /' | tee -a summary.txt
  grep -ah "P224_REF\|P224_FINGERPRINT" logs/ref_*.log | sed 's/^\[[0-9]*\] //' | tee -a summary.txt; }
# ---- six server processes: two ABBA blocks of P0 / PA / P0 (Amendment 3: no PA-DQ slot)
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json"
for spec in b0A1:P0:1 b0B:PA:1 b0A2:P0:0 b1A1:P0:0 b1B:PA:0 b1A2:P0:0; do
  IFS=: read -r TAG ARM SCORE <<< "$spec"
  can_run $((NEED_SLOT + SCORE * NEED_SCORE)) "slot $TAG" || break
  case "$ARM" in P0) AD="";; PA) AD=$W/run/adapter_REF;; esac
  phase_start "slot_$TAG" measure
  # shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
  env PYTHONPATH= $ENGINE_ENV ${AD:+E4B_PAGED_ADAPTER=$AD E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts} \
    perl -e "alarm $(alarm $SLOT_ALARM); exec @ARGV" python $W/p224_box.py --mode paged --arm $ARM --slot $TAG \
    $([ "$SCORE" = 1 ] && echo --score) --heldout $W/run/data/heldout.jsonl --http $W/run/data/http.json \
    --out $W/run/slot_$TAG.json > logs/slot_$TAG.log 2>&1
  echo "slot $TAG ($ARM) rc=$? $(grep -ah '^P224_COST' logs/slot_$TAG.log | cut -c1-300)" | tee -a summary.txt
  phase_end "slot_$TAG"; boundary "slot_$TAG"
done
# ---- collect and reduce
phase_start reduce reduce
python $W/p224_box.py --mode collect --dir $W/run | tee -a summary.txt
python - "$E4B_SHA" "$REV" "$CANARY_SHA256" > $W/run/pins.json <<'PY'
import json, sys
print(json.dumps({"e4b_sha": sys.argv[1], "model_revision": sys.argv[2], "canary_sha256": sys.argv[3]}))
PY
# P225's adapter gate on the real files, then P225's verdict over P224's reducer (PREREG "Outcomes")
python $W/p225_gate.py --run $W/run --revision $REV --out $W/run/gates_p225.json 2>&1 | tee -a summary.txt
python $W/p225_reduce.py --dir $W/run --pins $W/run/pins.json --gates $W/run/gates_p225.json --out $W/run/verdict_p225.json 2>&1 | tee -a summary.txt
phase_end reduce
V=NONE; [ -s $W/run/verdict_p225.json ] && V=$(python -c "import json; print(json.load(open('$W/run/verdict_p225.json'))['verdict'])")
[ "$V" != NONE ] || { say "REDUCER FAILED"; finish 22; }
case "$V" in PASS|FAIL|INCONCLUSIVE) finish 0;; *) say "P225 $V -- no reading"; finish 27;; esac
