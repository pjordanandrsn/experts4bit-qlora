#!/bin/bash
# Lane tp2 (P40: e4b vs Unsloth per MoE family, one RTX 5090, one fixture) -- pre-registered in tp2/P40-PREREG.md.
# e4b = the SHIPPED cut from PyPI in the image python (experts4bit-qlora 0.35.1 + grouped-nf4-gemm 0.30.2; transformers/bnb at tp1's
# pins, recorded); Unsloth = latest PyPI at launch in its own venv, NO transformers/bnb/peft pins (P38 amendment 1), torchao removed if
# the tripwire fails on ScalingType (P38 amendment 2). No usercustomize hook: tp1 proved a training lane needs none (serving K8 only).
# Pattern: p38_run.sh (per-arm `perl -e 'alarm N'`, summary.txt, TP_DONE, `say`, train anchor, dataset sha, helpers from the archive
# tarball) + tp1_run.sh (per-family fetch / arms / free, per-family alarms = tp1's + 50 %). Per family: fetch UNPINNED, assert the
# staged sha == the pinned revision, write refs/main from the pin only after that proof (P38 amendment 2 / e4b#404), tokenise once,
# arms in P40's order, free. A refusal / OOM / load fault / alarm is a row; nothing is coerced.
set -uo pipefail
LANE=tp2; W=/root/$LANE; mkdir -p $W $W/logs $W/adapters; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
E4B_VER=${TP2_E4B_VER:-0.35.1}; GNF4_VER=${TP2_GNF4_VER:-0.30.2}                 # P40 "Environments": the shipped cut, from PyPI
TF_VER=${TP2_TRANSFORMERS_VER:-5.16.1}; BNB_VER=${TP2_BNB_VER:-0.50.1}             # tp1's/P38's e4b-side pins (recorded in every receipt)
E4B_SRC_REF=${TP2_E4B_SRC_REF:-v$E4B_VER}                                            # the archive tarball the helpers come from (tag or sha)
STEPS=${TP2_STEPS:-60}; SEQ=${TP2_SEQ:-512}; EVAL_EVERY=${TP2_EVAL_EVERY:-20}; EVAL_N=${TP2_EVAL_N:-48}
LR=${TP2_LR:-1e-4}; ACCUM=${TP2_ACCUM:-1}; AUTOCAST=${TP2_AUTOCAST:-0}              # P40 amendment 1 (2026-09-05 23:00Z): P38's fixture as run — lr 1e-4, batch 1, accum 1, bf16 compute, no autocast context
R=${TP2_R:-8}; ALPHA=${TP2_ALPHA:-16}; SEED=${TP2_SEED:-0}
DATASET=${TP2_DATASET:-clinical}                                                     # the registered text; anything else is an amendment
ANCHOR_STRICT=${TP2_ANCHOR_STRICT:-1}
SKIP=${TP2_SKIP:-}                                                                   # family tags and/or fam/fw/tag tokens to skip
PIN_FALLBACK=${TP2_PIN_FALLBACK:-0}                                                  # 1 = on a pin mismatch fetch the pinned revision + refs/main := pin (an amendment)
UNS_LOADER=${TP2_UNSLOTH_LOADER:-FastLanguageModel}                                  # P38's loader; FastModel is an amendment
export TP2_INSTANCE_ID=${TP2_INSTANCE_ID:-$(cat /root/tp2/INSTANCE_ID 2>/dev/null)}
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
: > summary.txt
for f in tp2_arm.py tp2_reduce.py; do [ -s $W/$f ] || { echo "STAGE MISSING: $f"; touch TP_DONE; exit 9; }; done
echo "FIXTURE steps=$STEPS seq=$SEQ eval_every=$EVAL_EVERY eval_n=$EVAL_N lr=$LR accum=$ACCUM autocast=$AUTOCAST r=$R alpha=$ALPHA seed=$SEED dataset=$DATASET" | tee -a summary.txt
# ---------------------------------------------------------------- e4b install from PyPI (image python) + tripwire (versions by value, symbols by import, no hook)
say "install e4b (image python, PyPI): experts4bit-qlora==$E4B_VER grouped-nf4-gemm==$GNF4_VER transformers==$TF_VER bitsandbytes==$BNB_VER"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "experts4bit-qlora==$E4B_VER" "grouped-nf4-gemm==$GNF4_VER" "transformers==$TF_VER" "bitsandbytes==$BNB_VER" \
  datasets accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken > logs/pip_e4b.log 2>&1
rc=$?; echo "pip(e4b) rc=$rc"; [ $rc -ne 0 ] && { tail -4 logs/pip_e4b.log; echo "PIP FAIL (e4b)"; touch TP_DONE; exit 9; }
E4B_VER="$E4B_VER" GNF4_VER="$GNF4_VER" TF_VER="$TF_VER" python - <<'PYT' || { echo "TRIPWIRE FAIL (e4b)"; touch TP_DONE; exit 9; }
import importlib.metadata as md, inspect, os, sys
import experts4bit_qlora as e, torch, triton, transformers, bitsandbytes
from experts4bit_qlora import enable_fast_train, enable_batched_train, ExpertsLoRA, load_moe_4bit_streaming, verify_moe_4bit, disable_fast_train, disable_batched_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit   # TRAIN_ATTN_4BIT's mechanism (e4b#299)
from experts4bit_qlora.train import save_adapter
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated
from nf4_qlora import fused_grouped_lora
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
assert e.__version__ == os.environ["E4B_VER"], (e.__version__, os.environ["E4B_VER"])
assert md.version("experts4bit-qlora") == os.environ["E4B_VER"], md.version("experts4bit-qlora")
assert md.version("grouped-nf4-gemm") == os.environ["GNF4_VER"], md.version("grouped-nf4-gemm")
assert transformers.__version__ == os.environ["TF_VER"], transformers.__version__
try:
    import usercustomize; hook = getattr(usercustomize, "__file__", "?")
except ImportError:
    hook = None
assert hook is None or "/root/tp2" not in str(hook), f"a usercustomize hook is loaded from {hook} (a training lane needs none)"
print("tp2 tripwire OK (e4b):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "triton", triton.__version__,
      "transformers", transformers.__version__, "bnb", bitsandbytes.__version__, "usercustomize hook:", hook or "none (training lane; tp1 needed none)")
open("/root/tp2/versions.txt", "a").write(f"e4b {e.__version__} (PyPI)\ngnf4 {md.version('grouped-nf4-gemm')} (PyPI)\ntorch(e4b) {torch.__version__}\ntriton(e4b) {triton.__version__}\n"
                                          f"transformers(e4b) {transformers.__version__}\nbitsandbytes(e4b) {bitsandbytes.__version__}\nusercustomize_hook {hook or 'none'}\n")
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|NUMA node\(s\)" | tee -a forensics.txt; grep MemTotal /proc/meminfo | tee -a forensics.txt; cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX"; touch TP_DONE; exit 10; }
[ -s /root/.cache/huggingface/token ] || { echo "TOKEN MISSING (gemma-4-it and Mixtral-Instruct are gated)"; touch TP_DONE; exit 8; }
df -h /root | tail -1
# ---------------------------------------------------------------- Unsloth venv: latest PyPI at launch unless pinned; NO transformers/bnb/peft pins (P38 amendment 1)
UNS_VER=${TP2_UNSLOTH_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth/json',timeout=60))['info']['version'])" 2>/dev/null)}
ZOO_VER=${TP2_UNSLOTH_ZOO_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth-zoo/json',timeout=60))['info']['version'])" 2>/dev/null)}
UNS_OK=1; [ -z "$UNS_VER" ] && { echo "UNSLOTH VERSION UNRESOLVED (set TP2_UNSLOTH_VERSION)"; UNS_OK=0; }
if [ "$UNS_OK" = "1" ]; then
  say "install unsloth[cu128-torch280]==$UNS_VER unsloth_zoo==$ZOO_VER into $W/venv-unsloth (no transformers/bnb/peft pins)"
  python -m venv $W/venv-unsloth && perl -e 'alarm 2700; exec @ARGV' $W/venv-unsloth/bin/pip install -q --no-input --no-cache-dir \
    "unsloth[cu128-torch280]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.log 2>&1
  rc=$?; echo "pip(unsloth) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_unsloth.log; echo "PIP FAIL (unsloth) -- e4b arms still run; NO COMPARATOR"; UNS_OK=0; }
fi
cat > $W/tripwire_unsloth.py <<'PYU'
import importlib.metadata as md, torch, transformers, bitsandbytes, peft
import unsloth, unsloth_zoo
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches.common import is_transformers_v5_moe_quantization_available
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit, _is_bnb4bit_param, _moe_uses_bnb4bit_expert_weights
from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend, _should_use_separated_lora
assert is_transformers_v5_moe_quantization_available(), "the transformers-v5 4-bit MoE path is NOT available in this environment: Unsloth would not train 4-bit MoE here"
tri = None
try:
    import triton; tri = triton.__version__
except Exception: pass
tao = None
try:
    tao = md.version("torchao")
except Exception: pass
print("tp2 tripwire OK (unsloth):", unsloth.__version__, "zoo", unsloth_zoo.__version__, "torch", torch.__version__, "triton", tri, "transformers", transformers.__version__,
      "bnb", bitsandbytes.__version__, "peft", peft.__version__, "torchao", tao, "moe_backend", select_moe_backend(), "separated_lora", _should_use_separated_lora())
open("/root/tp2/versions.txt", "a").write(f"unsloth {unsloth.__version__}\nunsloth_zoo {unsloth_zoo.__version__}\ntorch(unsloth) {torch.__version__}\ntriton(unsloth) {tri}\n"
                                          f"transformers(unsloth) {transformers.__version__}\nbitsandbytes(unsloth) {bitsandbytes.__version__}\npeft {peft.__version__}\ntorchao {tao}\nmoe_backend {select_moe_backend()}\n")
PYU
if [ "$UNS_OK" = "1" ]; then
  $W/venv-unsloth/bin/python $W/tripwire_unsloth.py > logs/tripwire_unsloth.log 2>&1; trc=$?
  if [ $trc -ne 0 ] && grep -q "ScalingType" logs/tripwire_unsloth.log; then
    # P38 amendment 2: torchao (an optional import on transformers' lazy path) needs torch>=2.9 symbols; remove it and re-run the tripwire once
    say "tripwire(unsloth) failed on ScalingType -> removing torchao (P38 amendment 2)"; cp logs/tripwire_unsloth.log logs/tripwire_unsloth.attempt1.log
    $W/venv-unsloth/bin/pip uninstall -y -q torchao > logs/pip_unsloth_torchao_removed.log 2>&1
    echo "AMENDMENT-CLASS (P38 amendment 2): torchao removed from venv-unsloth after the tripwire failed on ScalingType" | tee -a summary.txt
    $W/venv-unsloth/bin/python $W/tripwire_unsloth.py > logs/tripwire_unsloth.log 2>&1; trc=$?
  fi
  [ $trc -ne 0 ] && { echo "TRIPWIRE FAIL (unsloth) -- e4b arms still run; NO COMPARATOR"; tail -5 logs/tripwire_unsloth.log; UNS_OK=0; }
  tail -1 logs/tripwire_unsloth.log
fi
# ---------------------------------------------------------------- repo helpers at the cut (archive tarball: tp1 amendment 2), box class, the fixed text
case "$E4B_SRC_REF" in *[!0-9a-f]*|"") SRC_URL="https://github.com/pjordanandrsn/experts4bit-qlora/archive/refs/tags/$E4B_SRC_REF.tar.gz";; *) SRC_URL="https://github.com/pjordanandrsn/experts4bit-qlora/archive/$E4B_SRC_REF.tar.gz";; esac
say "fetching repo helpers from $SRC_URL"
rm -rf $W/e4b-src && mkdir -p $W/e4b-src && perl -e 'alarm 600; exec @ARGV' bash -c "curl -sL $SRC_URL | tar xz -C $W/e4b-src --strip-components=1"; rc=$?
[ $rc -ne 0 ] || [ ! -s $W/e4b-src/bench/flagship-matrix/ds_manifest.json ] && { echo "SRC FETCH FAIL ($SRC_URL)"; touch TP_DONE; exit 9; }
cp $W/e4b-src/bench/flagship-matrix/drivers/n9_datasets.py $W/e4b-src/bench/flagship-matrix/ds_manifest.json $W/e4b-src/bench/train-anchor/train_anchor.py $W/e4b-src/bench/train-anchor/train_anchor_gate.py $W/
echo "HELPERS e4b-src ref=$E4B_SRC_REF ds_manifest sha=$(sha256sum $W/ds_manifest.json | awk '{print $1}')" | tee -a summary.txt
say "train anchor"
ANCHOR_OUT=$W/anchor.json perl -e 'alarm 900; exec @ARGV' python $W/train_anchor.py > logs/anchor.log 2>&1; tail -3 logs/anchor.log
python $W/train_anchor_gate.py $W/anchor.json | tee logs/anchor_gate.log; arc=${PIPESTATUS[0]}
export TP2_ANCHOR_JSON=$W/anchor.json TP2_BOX_CLASS="$(grep -E '^\s*class ' logs/anchor_gate.log | awk '{print $2}')"
echo "ANCHOR rc=$arc class=$TP2_BOX_CLASS" | tee -a summary.txt
if [ "$arc" -ne 0 ] && [ "$ANCHOR_STRICT" = "1" ]; then echo "BOX REFUSED by train anchor (rc=$arc)"; touch BOX_REFUSED TP_DONE; exit 12; fi
say "dataset: $DATASET (n9_datasets.py, sha-verified against ds_manifest.json)"
mkdir -p $W/data && (cd $W/data && python $W/n9_datasets.py $W/data > $W/logs/datasets.log 2>&1); tail -2 logs/datasets.log
DATA=$W/data/ds_$DATASET.json
DATA_SHA=$(python -c "import json; print(json.load(open('$W/ds_manifest.json'))['$DATASET']['sha256'])")
GOT_SHA=$(sha256sum $DATA | awk '{print $1}'); [ "$GOT_SHA" = "$DATA_SHA" ] || { echo "DATASET MISMATCH: $GOT_SHA != $DATA_SHA"; touch TP_DONE; exit 13; }
echo "DATASET $DATASET sha=$DATA_SHA" | tee -a summary.txt
# ---------------------------------------------------------------- helpers
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
# stubw FAM FW TAG ARM STATUS REASON [extra-json]: a row for an attempt that never reached the harness (the vocabulary's not_run / refused / load_fault / install_failed / alarm)
stubw(){ python - "$W" "$STEPS" "$SEQ" "$ACCUM" "$@" <<'PYS'
import json, os, sys
W, steps, seq, accum, fam, fw, tag, arm, status, reason = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), *sys.argv[5:11]
extra = json.loads(sys.argv[11]) if len(sys.argv) > 11 else {}
rec = {"framework": fw, "fam": fam, "arm": arm, "tag": tag, "status": status, "reason": reason[:800], "steps": steps, "seq": seq, "accum": accum,
       "written_by": "tp2_run.sh", "prereg": "tp2/P40-PREREG.md", **extra}
p = os.path.join(W, f"{fam}_{fw}_{tag}.json")
json.dump(rec, open(p, "w"), indent=1)
print(f"STUB {status.upper()} {fam}/{fw}/{tag}: {reason[:160]}")
PYS
  echo "$1/$2/$3 STUB $5: $6" | cut -c1-300 >> summary.txt; }
# fetch FAM MID REV ALARM: the snapshot UNPINNED (the loader resolves `main`, e4b#404), then the PROOF main == pin; refs/main written from the pin only then
fetch(){ local FAM=$1 MID=$2 REV=$3 AL=$4; say "fetch $FAM ($MID, unpinned; pin $REV)"
  perl -e "alarm $AL; exec @ARGV" python - "$MID" <<'PYF' > logs/fetch_$FAM.log 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
  local rc=$?; tail -2 logs/fetch_$FAM.log | head -1
  [ $rc -ne 0 ] && { echo "$FAM: FETCH FAILED rc=$rc" | tee -a summary.txt; FETCH_REASON="fetch failed rc=$rc (alarm $AL s; logs/fetch_$FAM.log)"; return 1; }
  local GOT; GOT=$(grep -a "^STAGED " logs/fetch_$FAM.log | tail -1 | awk '{print $2}')
  local RDIR=/root/.cache/huggingface/hub/models--${MID//\//--}
  if [ "$GOT" = "$REV" ]; then
    mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main
    [ "$(cat $RDIR/refs/main)" = "$REV" ] || { echo "$FAM: REFS/MAIN WRITE FAILED" | tee -a summary.txt; FETCH_REASON="refs/main could not be written"; return 1; }
    echo "PIN OK $FAM staged=$GOT == pin; refs/main written from the pin (P38 amendment 2 / e4b#404)" | tee -a summary.txt
  elif [ "$PIN_FALLBACK" = "1" ]; then
    say "PIN MISMATCH $FAM: staged $GOT != pin $REV -> fetching the pinned revision (TP2_PIN_FALLBACK=1, an amendment)"
    perl -e "alarm $AL; exec @ARGV" python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MID', revision='$REV', allow_patterns=['*.safetensors', '*.json', 'tokenizer*', '*.model', '*.txt', 'merges.txt', 'vocab.json'], max_workers=4))" > logs/fetch_${FAM}_pinned.log 2>&1 \
      || { echo "$FAM: PINNED FETCH FAILED" | tee -a summary.txt; FETCH_REASON="main moved past the pin ($GOT != $REV) and the pinned fetch failed"; return 1; }
    mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main
    echo "AMENDMENT (TP2_PIN_FALLBACK): $FAM main=$GOT != pin=$REV; pinned snapshot fetched; refs/main := pin (cache pointer only)" | tee -a summary.txt
  else
    echo "PIN MISMATCH $FAM: staged=$GOT != pin=$REV -- main moved past the pin; the family is ABORTED with load_fault stubs (not coerced)" | tee -a summary.txt
    FETCH_REASON="staged snapshot $GOT != pinned revision $REV (main moved past the pin); family aborted, not coerced (set TP2_PIN_FALLBACK=1 as an amendment to fetch the pin)"
    return 2
  fi
  df -h /root | tail -1; return 0; }
# expect_of FAM: the family's e4b trainable count (primary receipt first, then attn_only, then reference) for --expect-trainable
expect_of(){ python - "$W" "$1" <<'PYE' 2>/dev/null
import json, os, sys
W, fam = sys.argv[1], sys.argv[2]
for tag in ("fused_attn4", "attn_only", "reference_attn4"):
    p = os.path.join(W, f"{fam}_e4b_{tag}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        if r.get("status") == "ok" and r.get("trainable_params"):
            print(r["trainable_params"]); break
PYE
}
# arm FAM FW TAG ARM ALARM MID REV OFFLOAD [extra args...]: one process, one JSON (<FAM>_<FW>_<TAG>.json), one alarm; result line -> summary.txt
arm(){ local FAM=$1 FW=$2 TAG=$3 ARM=$4 AL=$5 MID=$6 REV=$7 OFF=$8; shift 8
  { skip $FAM || skip $FAM/$FW/$TAG; } && { say "skip $FAM/$FW/$TAG"; stubw $FAM $FW $TAG $ARM not_run "skipped by TP2_SKIP"; return 0; }
  local PY=python; [ "$FW" = "unsloth" ] && PY=$W/venv-unsloth/bin/python
  if [ "$FW" = "unsloth" ] && [ "$UNS_OK" != "1" ]; then stubw $FAM $FW $TAG $ARM install_failed "unsloth venv did not install/import (see logs/pip_unsloth.log, logs/tripwire_unsloth.log)"; return 0; fi
  local EXP; EXP=$(expect_of $FAM); local EXPARG=""; [ -n "$EXP" ] && EXPARG="--expect-trainable $EXP"
  say "arm $FAM/$FW/$TAG (arm=$ARM steps=$STEPS seq=$SEQ accum=$ACCUM autocast=$AUTOCAST lr=$LR eval_every=$EVAL_EVERY offload=$OFF alarm=$AL expect_trainable=${EXP:-none} $*)"
  local sp; sp=$(vram_start ${FAM}_${FW}_$TAG)
  HF_HUB_OFFLINE=1 UNSLOTH_ENABLE_LOGGING=1 perl -e "alarm $AL; exec @ARGV" $PY -u $W/tp2_arm.py --framework $FW --arm $ARM --tag $TAG --fam $FAM --model "$MID" --revision $REV \
      --steps $STEPS --seq $SEQ --accum $ACCUM --autocast $AUTOCAST --lr $LR --r $R --alpha $ALPHA --seed $SEED --offload $OFF \
      --tokens $TOK --tokens-sha $TOK_SHA --eval-every $EVAL_EVERY --eval-n $EVAL_N --unsloth-loader $UNS_LOADER $EXPARG \
      --out $W --adapter-dir $W/adapters "$@" > logs/run_${FAM}_${FW}_$TAG.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/${FAM}_${FW}_$TAG.json ]; then stubw $FAM $FW $TAG $ARM alarm "arm alarm $AL s (SIGALRM; the process could not write its own stub)"; fi
  grep -aE "^CELL |^LOAD OK|^ENGAGE|^STUB|Enabling LoRA on MoE|MoE bnb4bit|Error|error:" logs/run_${FAM}_${FW}_$TAG.log | tail -3 | cut -c1-300 | sed "s/^/    /"
  { echo -n "$FAM/$FW/$TAG rc=$rc "; grep -aE "^CELL " logs/run_${FAM}_${FW}_$TAG.log | tail -1 | cut -c1-400; echo; } >> summary.txt
  $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
free_family(){ rm -rf /root/.cache/huggingface/hub/models--$2; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
# family FAM MID REV FETCH_AL REF_AL FUSED_AL UNS_AL OFFLOAD MODE: P40's arms in P40's order; alarms = tp1's + 50 %; every attempt a row
family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 RAL=$5 FUAL=$6 UAL=$7 OFF=$8 MODE=$9
  if skip $FAM; then say "skip family $FAM (TP2_SKIP)"      # every attempt a row: the skipped arms are not_run stubs
    if [ "$MODE" = "gptoss" ]; then stubw $FAM e4b attn_only attn_only not_run "family skipped by TP2_SKIP"; else stubw $FAM e4b reference_attn4 reference not_run "family skipped by TP2_SKIP"; stubw $FAM e4b fused_attn4 fused not_run "family skipped by TP2_SKIP"; fi
    stubw $FAM unsloth ckpt_unsloth unsloth not_run "family skipped by TP2_SKIP"; [ "$MODE" = "anchor" ] && stubw $FAM unsloth ckpt_hf unsloth not_run "family skipped by TP2_SKIP"
    return 0
  fi
  say "===== family $FAM ($MID @ $REV; mode=$MODE offload=$OFF)"
  FETCH_REASON=""
  fetch $FAM $MID $REV $FAL; local frc=$?
  if [ $frc -ne 0 ]; then
    local st=not_run; [ $frc -eq 2 ] && st=load_fault
    if [ "$MODE" = "gptoss" ]; then stubw $FAM e4b attn_only attn_only $st "$FETCH_REASON"; else stubw $FAM e4b reference_attn4 reference $st "$FETCH_REASON"; stubw $FAM e4b fused_attn4 fused $st "$FETCH_REASON"; fi
    stubw $FAM unsloth ckpt_unsloth unsloth $st "$FETCH_REASON"; [ "$MODE" = "anchor" ] && stubw $FAM unsloth ckpt_hf unsloth $st "$FETCH_REASON"
    free_family $FAM ${MID//\//--}; return 0
  fi
  say "tokenise once ($FAM: the checkpoint's tokenizer at $REV, seq $SEQ) -> tokens_$FAM.json"
  TOK=$W/tokens_$FAM.json
  HF_HUB_OFFLINE=1 python $W/tp2_arm.py --prepare --fam $FAM --model "$MID" --revision $REV --data $DATA --data-sha $DATA_SHA --seq $SEQ --eval-n $EVAL_N --tokens $TOK > logs/prepare_$FAM.log 2>&1
  if [ $? -ne 0 ]; then
    tail -3 logs/prepare_$FAM.log; echo "$FAM: TOKENS FAIL" | tee -a summary.txt
    local why="tokenise failed (logs/prepare_$FAM.log): $(tail -1 logs/prepare_$FAM.log | cut -c1-200)"
    if [ "$MODE" = "gptoss" ]; then stubw $FAM e4b attn_only attn_only harness_error "$why"; else stubw $FAM e4b reference_attn4 reference harness_error "$why"; stubw $FAM e4b fused_attn4 fused harness_error "$why"; fi
    stubw $FAM unsloth ckpt_unsloth unsloth harness_error "$why"; [ "$MODE" = "anchor" ] && stubw $FAM unsloth ckpt_hf unsloth harness_error "$why"
    free_family $FAM ${MID//\//--}; return 0
  fi
  TOK_SHA=$(python -c "import json; print(json.load(open('$TOK'))['sha256'])"); tail -1 logs/prepare_$FAM.log; echo "TOKENS $FAM sha=$TOK_SHA" | tee -a summary.txt
  if [ "$MODE" = "gptoss" ]; then
    # P40 arm 2 on gpt-oss: skipped as REFUSED with the tp1 row cited (never coerced through NF4/ExpertsLoRA); attn_only runs instead as the secondary row
    # and refreshes this stub with its own enable_fast_train probe on this box (tp2_arm.py T3).
    stubw $FAM e4b fused_attn4 fused refused "SKIPPED as REFUSED: tp1 (P36) row cited -- enable_fast_train(dgrad=True) patched 0 modules on gpt-oss (experts built bare, no ExpertsLoRA: 'GPT-OSS-aware training LoRA is a separate change', loader.py); P40 arm 2: not run, attn_only is the secondary row" '{"cited": "tp1", "n_patched": 0}'
    arm $FAM e4b attn_only attn_only $RAL "$MID" $REV $OFF --attn-4bit 0
  else
    arm $FAM e4b reference_attn4 reference $RAL "$MID" $REV $OFF --attn-4bit 1
    arm $FAM e4b fused_attn4 fused $FUAL "$MID" $REV $OFF --attn-4bit 1
  fi
  arm $FAM unsloth ckpt_unsloth unsloth $UAL "$MID" $REV 0 --grad-ckpt unsloth
  [ "$MODE" = "anchor" ] && arm $FAM unsloth ckpt_hf unsloth $UAL "$MID" $REV 0 --grad-ckpt hf
  echo "$(echo $FAM | tr a-z A-Z) DONE"
  free_family $FAM ${MID//\//--}; }
# ---------------------------------------------------------------- families in P40's order (small to large, Mixtral last); revisions = tp1's staged snapshots
#        FAM      MID                                            REV                                         FETCH REF  FUSED UNS  OFFLOAD MODE
family granite  ibm-granite/granite-3.1-3b-a800m-instruct       a02780686e08a03fe0d2679a293b5c74a90efa89    3600  1800 1800  1800 0 normal
family olmoe    allenai/OLMoE-1B-7B-0924-Instruct               7f1c97f440f06ce36705e4f2b843edb5925f4498    3600  2250 2250  2250 0 normal
family gptoss   openai/gpt-oss-20b                              6cee5e81ee83917806bbde320786a8fb61efebee    4500  3600 3600  3600 0 gptoss
family qwen3    Qwen/Qwen3-30B-A3B                              ad44e777bcd18fa416d9da3bd8f70d33ebb85d39    7200  4500 3600  4500 0 anchor
family gemma4   google/gemma-4-26B-A4B-it                       4d7ae4984b7db7de8f8457170b3f1a419ee76d52    7200  4500 3600  4500 0 normal
family mixtral  mistralai/Mixtral-8x7B-Instruct-v0.1            eba92302a2861cdc0098cc54bc9f17cb2c47eb61    10800 6300 5400  6300 1 normal
# ---------------------------------------------------------------- reduce, summarise, mark
say "reduce"
python $W/tp2_reduce.py $W --md $W/RESULTS-tp2.md --steps $STEPS | tee RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
[ "$UNS_OK" = "1" ] || echo "NO COMPARATOR: the Unsloth side did not install/import; e4b arms only -- not an H2H" | tee -a summary.txt
say "TP_DONE"; touch TP_DONE
