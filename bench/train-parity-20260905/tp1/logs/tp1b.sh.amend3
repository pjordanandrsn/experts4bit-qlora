#!/bin/bash
# tp1b (P36 amendment 3, 2026-09-05 11:55Z): re-run Granite's `fused` arm on the tp1 box after TP_DONE. The lane's first Granite
# fused arm died on a harness defect (the kernel-call counter's late-binding closure called _dequant_whole for fused_grouped_lora);
# the harness was patched on the box before OLMoE's fused arm. Same box, same install, same fixture; marker TP2_DONE.
set -uo pipefail
LANE=tp1; W=/root/$LANE; mkdir -p $W $W/logs; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
E4B_SHA=${E4B_SHA:-f4b639fd2640}; GNF4_SHA=${GNF4_SHA:-ddcb850e05c3}
STEPS=${TP1_STEPS:-60}; SEQ=${TP1_SEQ:-512}
DATASET=${TP1_DATASET:-clinical}          # the fused-train-gate's text; sha-verified against ds_manifest.json
ANCHOR_STRICT=${TP1_ANCHOR_STRICT:-1}      # 1 = a box the train anchor REFUSES ends the lane (exit 12) so the queue re-rents
MIXTRAL_RESIDENT_PROBE=${TP1_MIXTRAL_RESIDENT_PROBE:-0}
SKIP=${TP1_SKIP:-}                         # space-separated family tags to skip, e.g. "gemma4"
export TP1_INSTANCE_ID=${TP1_INSTANCE_ID:-$(cat /root/tp1/INSTANCE_ID 2>/dev/null)}   # written by tp1_all.sh at launch; joins every receipt to the bill
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
# ---------------------------------------------------------------- helpers
# fetch MID TAG ALARM: the snapshot, once per family (three arms share it)
fetch(){ local MID=$1 TAG=$2 AL=$3; say "fetch $TAG ($MID)"
  perl -e "alarm $AL; exec @ARGV" python - "$MID" <<'PYF' > logs/fetch_$TAG.log 2>&1
import sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"])
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
PYF
  rc=$?; tail -1 logs/fetch_$TAG.log; [ $rc -ne 0 ] && { echo "$TAG: FETCH FAILED rc=$rc" | tee -a summary.txt; return 1; }
  df -h /root | tail -1; return 0; }
# arm TAG ARM ALARM MID OFFLOAD [extra args...]: one process, one JSON (<TAG>_train_<ARM>.json), one alarm; result line -> summary.txt
# ARMOPT=<reference|fused|batched|attn_only> overrides the harness arm while ARM stays the JSON tag (the Mixtral resident probe).
arm(){ local TAG=$1 ARM=$2 AL=$3 MID=$4 OFF=$5; shift 5
  say "arm $TAG/$ARM (offload=$OFF steps=$STEPS seq=$SEQ alarm=$AL)"
  perl -e "alarm $AL; exec @ARGV" python -u $W/tp1_train_smoke.py --model "$MID" --fam $TAG --arm ${ARMOPT:-$ARM} --tag $ARM --steps $STEPS --seq $SEQ \
      --offload $OFF --data $DATA --data-sha $DATA_SHA --out $W "$@" > logs/run_${TAG}_$ARM.log 2>&1
  local rc=$?
  if [ $rc -eq 142 ] && [ ! -s $W/${TAG}_train_$ARM.json ]; then   # SIGALRM: the process cannot write its own stub
    python -c "import json; json.dump({'fam':'$TAG','arm':'$ARM','status':'alarm','reason':'arm alarm $AL s','n_patched':0,'steps':$STEPS}, open('$W/${TAG}_train_$ARM.json','w'), indent=1)"
  fi
  grep -aE "^CELL |^LOAD OK|Error|error:" logs/run_${TAG}_$ARM.log | tail -2 | sed "s/^/    /"
  { echo -n "$TAG/$ARM rc=$rc "; grep -aE "^CELL " logs/run_${TAG}_$ARM.log | tail -1 | cut -c1-400; } >> summary.txt
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
free_family(){ rm -rf /root/.cache/huggingface/hub/models--$2; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
# ---------------------------------------------------------------- families, smallest first, frees between
# Granite-3.1-3B-A800M: first real-weight pass through the direct loader + ExpertsLoRA + both accelerators (resident)
GR=ibm-granite/granite-3.1-3b-a800m-instruct
say "tp1b: Granite fused arm re-run on the patched harness"
python3 -c "import ast,sys; t=open('$W/tp1_train_smoke.py').read(); assert '_orig=orig' in t, 'harness not patched'; print('harness patched OK')" || { echo 'TP1B HARNESS UNPATCHED'; touch TP2_DONE; exit 9; }
GR=ibm-granite/granite-3.1-3b-a800m-instruct
fetch $GR granite 2400 && { arm granite fused 1200 $GR 0; echo "GRANITE FUSED RERUN DONE"; }
free_family granite ibm-granite--granite-3.1-3b-a800m-instruct 2>/dev/null || true
say "TP1B DONE"; touch TP2_DONE
