#!/bin/bash
# p38c (P38 amendment 2, 2026-09-05 16:30Z): the registered arms re-run on the same box after two environment defects, neither a
# workload change. (a) e4b arms: the lane's pinned snapshot_download(revision=<sha>) wrote no refs/main, and the e4b loader resolves
# the model id at 'main' (loader.py:590, AutoConfig without revision) -> LocalEntryNotFoundError under HF_HUB_OFFLINE=1 on every e4b arm.
# Fix: refs/main := the pinned sha (tp1 staged the same snapshot ad44e777 unpinned, so main == the pin at fetch time); the bytes are
# unchanged. (b) Unsloth venv: unsloth[cu128-torch280] resolved torchao 0.18.0, which imports torch>=2.9 symbols (ScalingType) -> the
# tripwire failed at 'import peft'. Fix: torchao removed from the venv (an optional import reached through transformers' lazy module);
# the tripwire then passes (unsloth 2026.9.2, zoo 2026.9.1, transformers 5.5.0, bnb 0.50.2, peft 0.20.0, torch 2.8.0+cu128).
# The failed rows stay in summary.txt above this run's lines; the install_failed stubs move to attempts/. Marker TP3_DONE.
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
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
cd $W || exit 1
R=/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B; [ "$(cat $R/refs/main 2>/dev/null)" = "$REV" ] || { echo "refs/main != $REV"; exit 9; }
echo "AMENDMENT 2: refs/main=$REV (cache pointer only; tp1 staged the same snapshot unpinned); torchao removed from venv-unsloth (0.18.0 needs torch>=2.9)" | tee -a summary.txt
mkdir -p attempts; for f in *_train_*.attempt1.json; do [ -e "$f" ] && mv "$f" attempts/; done; ls attempts
UNS_OK=1; venv-unsloth/bin/python tripwire.py > logs/tripwire_unsloth.log 2>&1 || { echo 'TRIPWIRE FAIL (unsloth)'; UNS_OK=0; }; tail -1 logs/tripwire_unsloth.log
TOK=$(ls $W/tokens_*.json | head -1); TOK_SHA=$(python -c "import json; print(json.load(open('$TOK'))['sha256'])"); echo "TOKENS(p38c) $TOK sha=$TOK_SHA" | tee -a summary.txt
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
say "p38c: registered arms, registered order (amendment 2)"
arm e4b     reference_attn4        reference  $STEPS       3000 --attn-4bit 1
arm e4b     fused_attn4            fused      $STEPS       2400 --attn-4bit 1
arm e4b     fused                  fused      $STEPS       2400 --attn-4bit 0
arm e4b     fused_attn4_nosamp     fused      20           1200 --attn-4bit 1 --no-sampler 1
arm unsloth ckpt_unsloth           unsloth    $STEPS       2400 --grad-ckpt unsloth
arm unsloth ckpt_hf                unsloth    $STEPS       2400 --grad-ckpt hf
arm e4b     fused_attn4_200        fused      $STEPS_CURVE 4200 --attn-4bit 1
arm unsloth ckpt_unsloth_200       unsloth    $STEPS_CURVE 4200 --grad-ckpt unsloth
say "reduce (p38c)"; python $W/p38_reduce.py $W --md $W/RESULTS-p38.md --target $TARGET | tee RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt 2>/dev/null
say "P38C DONE"; touch TP3_DONE
