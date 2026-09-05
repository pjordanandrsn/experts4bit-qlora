#!/bin/bash
# p38b (P38 amendment 1, 2026-09-05 16:05Z): the Unsloth comparator arms, re-run after the lane's TP_DONE on the same box. The lane's
# Unsloth venv install failed on a dependency conflict (it pinned transformers==5.16.1 next to unsloth, which requires its own range),
# so the lane ran e4b-only. Here the venv takes Unsloth's own resolution (versions recorded in versions.txt); the e4b side is untouched.
# The install_failed stubs are kept as *.attempt1.json; these arms are the rows that count. Marker TP2_DONE.
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
UNS_VER=${P38_UNSLOTH_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth/json',timeout=60))['info']['version'])" 2>/dev/null)}
ZOO_VER=${P38_UNSLOTH_ZOO_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth-zoo/json',timeout=60))['info']['version'])" 2>/dev/null)}
UNS_OK=1; [ -z "$UNS_VER" ] && { echo "UNSLOTH VERSION UNRESOLVED"; UNS_OK=0; }
if [ "$UNS_OK" = "1" ]; then
  say "install (amendment 1: no transformers/bnb/peft pins) unsloth[cu128-torch280]==$UNS_VER unsloth_zoo==$ZOO_VER into $W/venv-unsloth"
  rm -rf $W/venv-unsloth; python -m venv $W/venv-unsloth && perl -e 'alarm 2700; exec @ARGV' $W/venv-unsloth/bin/pip install -q --no-input --no-cache-dir \
    "unsloth[cu128-torch280]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.attempt2.log 2>&1
  rc=$?; echo "pip(unsloth, attempt 2) rc=$rc"; [ $rc -ne 0 ] && { tail -8 logs/pip_unsloth.attempt2.log; echo "PIP FAIL (unsloth, attempt 2) -- NO COMPARATOR"; UNS_OK=0; }
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
[ "$UNS_OK" = "1" ] || { echo "P38B: NO COMPARATOR (see logs/pip_unsloth.attempt2.log / logs/tripwire_unsloth.log)"; touch TP2_DONE; exit 9; }
TOK=$(ls $W/tokens_*.json | head -1); TOK_SHA=$(python -c "import json; print(json.load(open('$TOK'))['sha256'])"); echo "TOKENS(p38b) $TOK sha=$TOK_SHA" | tee -a summary.txt
for t in ckpt_unsloth ckpt_hf ckpt_unsloth_200; do [ -s $W/unsloth_train_$t.json ] && mv $W/unsloth_train_$t.json $W/unsloth_train_$t.attempt1.json; done
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
say "p38b: Unsloth comparator arms (amendment 1)"
arm unsloth ckpt_unsloth           unsloth    $STEPS       2400 --grad-ckpt unsloth
arm unsloth ckpt_hf                unsloth    $STEPS       2400 --grad-ckpt hf
arm unsloth ckpt_unsloth_200       unsloth    $STEPS_CURVE 4200 --grad-ckpt unsloth
say "reduce (p38b)"; python $W/p38_reduce.py $W --md $W/RESULTS-p38.md --target $TARGET | tee RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
say "P38B DONE"; touch TP2_DONE
