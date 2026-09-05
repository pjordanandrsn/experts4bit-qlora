#!/bin/bash
# Lane p38 (e4b vs Unsloth QLoRA end-to-end, same box, one training problem) -- pre-registered in P38-UNSLOTH-PREREG.md.
# Cut = the shipped code: e4b main @f4b639fd2640 (= v0.35.0 + #396 docs) + gnf4 main @ddcb850e05c3 (= v0.30.0 + #341 docs) in the IMAGE python
# (tp1's environment: torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1, bnb 0.50.1); Unsloth = latest PyPI at launch in its own venv.
# Pattern: tp1_run.sh (per-arm `perl -e 'alarm N'`, summary.txt, TP_DONE, `say`, train anchor, dataset sha, helpers from the archive tarball).
set -uo pipefail
LANE=p38; W=/root/$LANE; mkdir -p $W $W/logs $W/adapters; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
E4B_SHA=${E4B_SHA:-f4b639fd2640bb603b6b7b63ea010312b0bb351d}; GNF4_SHA=${GNF4_SHA:-ddcb850e05c3595e2bb87813df70427f7e4bafce}
MID=Qwen/Qwen3-30B-A3B; REV=${P38_REV:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}
STEPS=${P38_STEPS:-60}; STEPS_CURVE=${P38_STEPS_CURVE:-200}; SEQ=${P38_SEQ:-512}; TARGET=${P38_TARGET_LOSS:-0.32}; EVAL_EVERY=${P38_EVAL_EVERY:-20}
DATASET=${P38_DATASET:-clinical}           # `alpaca` = the registered alternative slice (an amendment if used)
ANCHOR_STRICT=${P38_ANCHOR_STRICT:-1}
SKIP=${P38_SKIP:-}
export P38_INSTANCE_ID=${P38_INSTANCE_ID:-$(cat /root/p38/INSTANCE_ID 2>/dev/null)}
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
: > summary.txt
for f in p38_arm.py p38_reduce.py; do [ -s $W/$f ] || { echo "STAGE MISSING: $f"; touch TP_DONE; exit 9; }; done
# ---------------------------------------------------------------- e4b install (image python) + tripwire (tp1's, plus the attn-4bit symbol)
say "install e4b (image python): gnf4 @$GNF4_SHA + e4b @$E4B_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken > logs/pip_e4b.log 2>&1
rc=$?; echo "pip(e4b) rc=$rc"; [ $rc -ne 0 ] && { tail -4 logs/pip_e4b.log; echo "PIP FAIL (e4b)"; touch TP_DONE; exit 9; }
python - <<'PYT' || { echo "TRIPWIRE FAIL (e4b)"; touch TP_DONE; exit 9; }
import importlib.metadata as md, inspect, os
import experts4bit_qlora as e, torch, triton, transformers, bitsandbytes
from experts4bit_qlora import enable_fast_train, enable_batched_train, ExpertsLoRA, load_moe_4bit_streaming, verify_moe_4bit
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit   # TRAIN_ATTN_4BIT's mechanism (e4b#299)
from experts4bit_qlora.train import save_adapter
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated
from nf4_qlora import fused_grouped_lora
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
assert e.__version__ == "0.35.0", e.__version__
assert md.version("grouped-nf4-gemm") == "0.30.0", md.version("grouped-nf4-gemm")
assert transformers.__version__ == "5.16.1" and triton.__version__.startswith("3.4"), (transformers.__version__, triton.__version__)
print("p38 tripwire OK (e4b):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "triton", triton.__version__, "bnb", bitsandbytes.__version__)
open("/root/p38/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ.get('E4B_SHA','')}\ngnf4 {md.version('grouped-nf4-gemm')}\ntorch(e4b) {torch.__version__}\ntriton(e4b) {triton.__version__}\ntransformers(e4b) {transformers.__version__}\nbitsandbytes(e4b) {bitsandbytes.__version__}\n")
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|NUMA node\(s\)" | tee -a forensics.txt; grep MemTotal /proc/meminfo | tee -a forensics.txt; cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX"; touch TP_DONE; exit 10; }
[ -s /root/.cache/huggingface/token ] || echo "NOTE: no HF token staged (Qwen3 is not gated; rate limits apply)"
df -h /root | tail -1
# ---------------------------------------------------------------- Unsloth venv (latest PyPI at launch unless pinned) + tripwire (the 4-bit MoE path must be AVAILABLE)
UNS_VER=${P38_UNSLOTH_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth/json',timeout=60))['info']['version'])" 2>/dev/null)}
ZOO_VER=${P38_UNSLOTH_ZOO_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth-zoo/json',timeout=60))['info']['version'])" 2>/dev/null)}
UNS_OK=1; [ -z "$UNS_VER" ] && { echo "UNSLOTH VERSION UNRESOLVED (set P38_UNSLOTH_VERSION)"; UNS_OK=0; }
if [ "$UNS_OK" = "1" ]; then
  say "install unsloth[cu128-torch280]==$UNS_VER unsloth_zoo==$ZOO_VER into $W/venv-unsloth"
  python -m venv $W/venv-unsloth && perl -e 'alarm 2700; exec @ARGV' $W/venv-unsloth/bin/pip install -q --no-input --no-cache-dir \
    "unsloth[cu128-torch280]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} "transformers==5.16.1" "bitsandbytes==0.50.1" "peft==0.20.0" datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.log 2>&1
  rc=$?; echo "pip(unsloth) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_unsloth.log; echo "PIP FAIL (unsloth) -- e4b arms still run; NO COMPARATOR"; UNS_OK=0; }
fi
if [ "$UNS_OK" = "1" ]; then
  $W/venv-unsloth/bin/python - <<'PYU' > logs/tripwire_unsloth.log 2>&1 || { echo "TRIPWIRE FAIL (unsloth) -- e4b arms still run; NO COMPARATOR"; tail -5 logs/tripwire_unsloth.log; UNS_OK=0; }
import importlib.metadata as md, torch, transformers, bitsandbytes, peft
import unsloth, unsloth_zoo
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches.common import is_transformers_v5_moe_quantization_available
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit, _is_bnb4bit_param
from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend, _should_use_separated_lora
assert is_transformers_v5_moe_quantization_available(), "the transformers-v5 4-bit MoE path is NOT available in this environment: Unsloth would not train 4-bit MoE here"
tri = None
try:
    import triton; tri = triton.__version__
except Exception: pass
print("p38 tripwire OK (unsloth):", unsloth.__version__, "zoo", unsloth_zoo.__version__, "torch", torch.__version__, "triton", tri, "transformers", transformers.__version__,
      "bnb", bitsandbytes.__version__, "peft", peft.__version__, "moe_backend", select_moe_backend(), "separated_lora", _should_use_separated_lora())
open("/root/p38/versions.txt", "a").write(f"unsloth {unsloth.__version__}\nunsloth_zoo {unsloth_zoo.__version__}\ntorch(unsloth) {torch.__version__}\ntriton(unsloth) {tri}\ntransformers(unsloth) {transformers.__version__}\nbitsandbytes(unsloth) {bitsandbytes.__version__}\npeft {peft.__version__}\nmoe_backend {select_moe_backend()}\n")
PYU
  tail -1 logs/tripwire_unsloth.log
fi
# ---------------------------------------------------------------- repo helpers at the cut (archive tarball: tp1 amendment 2), box class, the fixed text
say "fetching repo helpers at e4b @$E4B_SHA (archive tarball)"
rm -rf $W/e4b-src && mkdir -p $W/e4b-src && perl -e 'alarm 600; exec @ARGV' bash -c "curl -sL https://github.com/pjordanandrsn/experts4bit-qlora/archive/$E4B_SHA.tar.gz | tar xz -C $W/e4b-src --strip-components=1"; rc=$?
[ $rc -ne 0 ] || [ ! -s $W/e4b-src/bench/flagship-matrix/ds_manifest.json ] && { echo "SRC FETCH FAIL"; touch TP_DONE; exit 9; }
cp $W/e4b-src/bench/flagship-matrix/drivers/n9_datasets.py $W/e4b-src/bench/flagship-matrix/ds_manifest.json $W/e4b-src/bench/train-anchor/train_anchor.py $W/e4b-src/bench/train-anchor/train_anchor_gate.py $W/
say "train anchor"
ANCHOR_OUT=$W/anchor.json perl -e 'alarm 900; exec @ARGV' python $W/train_anchor.py > logs/anchor.log 2>&1; tail -3 logs/anchor.log
python $W/train_anchor_gate.py $W/anchor.json | tee logs/anchor_gate.log; arc=${PIPESTATUS[0]}
export P38_ANCHOR_JSON=$W/anchor.json P38_BOX_CLASS="$(grep -E '^\s*class ' logs/anchor_gate.log | awk '{print $2}')"
echo "ANCHOR rc=$arc class=$P38_BOX_CLASS" | tee -a summary.txt
if [ "$arc" -ne 0 ] && [ "$ANCHOR_STRICT" = "1" ]; then echo "BOX REFUSED by train anchor (rc=$arc)"; touch BOX_REFUSED TP_DONE; exit 12; fi
say "dataset: $DATASET"
mkdir -p $W/data
if [ "$DATASET" = "clinical" ]; then
  (cd $W/data && python $W/n9_datasets.py $W/data > $W/logs/datasets.log 2>&1); tail -2 logs/datasets.log
  DATA=$W/data/ds_clinical.json
  DATA_SHA=$(python -c "import json; print(json.load(open('$W/ds_manifest.json'))['clinical']['sha256'])")
  GOT_SHA=$(sha256sum $DATA | awk '{print $1}'); [ "$GOT_SHA" = "$DATA_SHA" ] || { echo "DATASET MISMATCH: $GOT_SHA != $DATA_SHA"; touch TP_DONE; exit 13; }
else
  DATA=$W/data/ds_alpaca.json; python $W/p38_arm.py --make-alpaca $DATA > logs/datasets.log 2>&1 || { echo "ALPACA SLICE FAIL"; tail -3 logs/datasets.log; touch TP_DONE; exit 13; }
  DATA_SHA=$(sha256sum $DATA | awk '{print $1}'); echo "AMENDMENT: dataset=alpaca (registered alternative slice) sha=$DATA_SHA" | tee -a summary.txt
fi
echo "DATASET $DATASET sha=$DATA_SHA" | tee -a summary.txt
# ---------------------------------------------------------------- the checkpoint (revision PINNED, once), then tokenise ONCE -> tokens_<dataset>.json (both arms load this file)
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python - "$MID" "$REV" <<'PYF' > logs/fetch_qwen3.log 2>&1
import sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
PYF
rc=$?; tail -1 logs/fetch_qwen3.log; [ $rc -ne 0 ] && { echo "DL FAIL"; touch TP_DONE; exit 11; }
export HF_HUB_OFFLINE=1   # from here both pythons read the same cached snapshot bytes
say "tokenise once (the checkpoint's tokenizer at $REV, seq $SEQ) -> tokens_$DATASET.json"
TOK=$W/tokens_$DATASET.json
python $W/p38_arm.py --prepare --model $MID --revision $REV --data $DATA --data-sha $DATA_SHA --seq $SEQ --tokens $TOK > logs/prepare.log 2>&1 || { echo "TOKENS FAIL"; tail -3 logs/prepare.log; touch TP_DONE; exit 14; }
TOK_SHA=$(python -c "import json; print(json.load(open('$TOK'))['sha256'])"); tail -1 logs/prepare.log; echo "TOKENS $DATASET sha=$TOK_SHA" | tee -a summary.txt
# ---------------------------------------------------------------- helpers
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
# arm FW TAG ARM STEPS ALARM [extra args...]: one process, one JSON (<FW>_train_<TAG>.json), one alarm; result line -> summary.txt
arm(){ local FW=$1 TAG=$2 ARM=$3 N=$4 AL=$5; shift 5
  skip $TAG && { say "skip $FW/$TAG"; return 0; }
  local PY=python; [ "$FW" = "unsloth" ] && PY=$W/venv-unsloth/bin/python
  if [ "$FW" = "unsloth" ] && [ "$UNS_OK" != "1" ]; then python -c "import json; json.dump({'framework':'unsloth','arm':'$ARM','tag':'$TAG','status':'install_failed','reason':'unsloth venv did not install/import (see logs/pip_unsloth.log, logs/tripwire_unsloth.log)','steps':$N}, open('$W/unsloth_train_$TAG.json','w'), indent=1)"; echo "unsloth/$TAG INSTALL_FAILED" >> summary.txt; return 0; fi
  say "arm $FW/$TAG (arm=$ARM steps=$N seq=$SEQ eval_every=$EVAL_EVERY target=$TARGET alarm=$AL $*)"
  local sp; sp=$(vram_start ${FW}_$TAG)
  UNSLOTH_ENABLE_LOGGING=1 perl -e "alarm $AL; exec @ARGV" $PY -u $W/p38_arm.py --framework $FW --arm $ARM --tag $TAG --model $MID --revision $REV --steps $N --seq $SEQ \
      --tokens $TOK --tokens-sha $TOK_SHA --eval-every $EVAL_EVERY --target-loss $TARGET --out $W --adapter-dir $W/adapters "$@" > logs/run_${FW}_$TAG.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/${FW}_train_$TAG.json ]; then python -c "import json; json.dump({'framework':'$FW','arm':'$ARM','tag':'$TAG','status':'alarm','reason':'arm alarm $AL s','steps':$N}, open('$W/${FW}_train_$TAG.json','w'), indent=1)"; fi
  grep -aE "^CELL |^LOAD OK|^ENGAGE|Enabling LoRA on MoE|MoE bnb4bit|Error|error:" logs/run_${FW}_$TAG.log | tail -3 | cut -c1-300 | sed "s/^/    /"
  { echo -n "$FW/$TAG rc=$rc "; grep -aE "^CELL " logs/run_${FW}_$TAG.log | tail -1 | cut -c1-400; echo; } >> summary.txt
  $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
# ---------------------------------------------------------------- arms, in the pre-registered order
arm e4b     reference_attn4        reference  $STEPS       3000 --attn-4bit 1
arm e4b     fused_attn4            fused      $STEPS       2400 --attn-4bit 1
arm e4b     fused                  fused      $STEPS       2400 --attn-4bit 0
arm e4b     fused_attn4_nosamp     fused      20           1200 --attn-4bit 1 --no-sampler 1
arm unsloth ckpt_unsloth           unsloth    $STEPS       2400 --grad-ckpt unsloth
arm unsloth ckpt_hf                unsloth    $STEPS       2400 --grad-ckpt hf
arm e4b     fused_attn4_200        fused      $STEPS_CURVE 4200 --attn-4bit 1
arm unsloth ckpt_unsloth_200       unsloth    $STEPS_CURVE 4200 --grad-ckpt unsloth
# ---------------------------------------------------------------- reduce, summarise, mark
say "reduce"
python $W/p38_reduce.py $W --md $W/RESULTS-p38.md --target $TARGET | tee RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
[ "$UNS_OK" = "1" ] || echo "NO COMPARATOR: the Unsloth side did not install/import; e4b arms only -- not an H2H" | tee -a summary.txt
say "TP_DONE"; touch TP_DONE
