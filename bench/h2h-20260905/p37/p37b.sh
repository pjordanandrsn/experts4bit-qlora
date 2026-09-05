#!/bin/bash
# p37b (P37 amendment 2, 2026-09-05 16:25Z): the vLLM comparator arms, run after the lane's TP_DONE on the same box. The lane's GPTQ
# fetch hung on a resumed read (9.7 GB of the 17 GB shard, cache flat, threads in futex wait, no socket) and was left to its 1800-s
# alarm per the STALL rule (look, never kill) -> 'DL FAIL (gptq) -- NO COMPARATOR', e4b arms only. Here: resume the fetch (up to 3
# attempts, 2400 s each), then the eight registered vLLM arms in the registered order against the SAME prompts_b{1,16}.json, reduce,
# TP2_DONE. Nothing on the e4b side is touched; the lane's SKIPPED lines stay in summary.txt above this run's lines.
set -uo pipefail
LANE=p37; W=/root/$LANE; mkdir -p $W $W/logs $W/hook; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
E4B_SHA=${E4B_SHA:-f4b639fd2640bb603b6b7b63ea010312b0bb351d}; GNF4_SHA=${GNF4_SHA:-ddcb850e05c3595e2bb87813df70427f7e4bafce}
BF16_MID=Qwen/Qwen3-30B-A3B;            BF16_REV=${P37_BF16_REV:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}
GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4;  GPTQ_REV=${P37_GPTQ_REV:-9b534e4318b7ebc3c961a839f13eb18b1833f441}
E4B_TORCH_CU128=${P37_E4B_TORCH_CU128:-0}   # 1 = force-reinstall torch 2.8.0+cu128 into the image python (bo7's exact toolchain); default = the image's torch
SKIP=${P37_SKIP:-}                          # space-separated arm tags to skip (e.g. "lic_eager")
export P37_INSTANCE_ID=${P37_INSTANCE_ID:-$(cat /root/p37/INSTANCE_ID 2>/dev/null)}   # written by the launcher; joins every receipt to the bill
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
fetch(){ local MID=$1 REV=$2 TAG=$3 AL=$4; say "fetch $TAG ($MID @ $REV)"
  perl -e "alarm $AL; exec @ARGV" python - "$MID" "$REV" <<'PYF' > logs/fetch_$TAG.log 2>&1
import sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
PYF
  rc=$?; tail -1 logs/fetch_$TAG.log; [ $rc -ne 0 ] && { echo "$TAG: DL FAIL rc=$rc" | tee -a summary.txt; return 1; }; df -h /root | tail -1; return 0; }
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
fenv(){ case "$1" in 0) echo "0 0 0";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
vllm_arm(){ local ARM=$1 B=$2; [ "$VLLM_OK" = "1" ] || { echo "vllm/$ARM B=$B SKIPPED (no comparator)" >> summary.txt; return 0; }
  skip $ARM && { say "skip vllm/$ARM B=$B"; return 0; }
  say "arm vllm/$ARM (B=$B vllm==$VLLM_VER alarm=2400 prompts_sha=$([ $B = 1 ] && echo $PSHA1 || echo $PSHA16))"
  local sp; sp=$(vram_start vllm_b${B}_$ARM)
  env P37_ARM=$ARM P37_BATCH=$B P37_PROMPTS=$W/prompts_b$B.json P37_MODEL=$GPTQ_MID P37_REV=$GPTQ_REV P37_OUT=$W/vllm_b${B}_$ARM.json VLLM_LOGGING_LEVEL=INFO \
    perl -e 'alarm 2400; exec @ARGV' $W/venv-vllm/bin/python $W/p37_vllm.py > logs/run_vllm_b${B}_$ARM.log 2>&1
  local rc=$?; vram_stop $sp
  if [ ! -s $W/vllm_b${B}_$ARM.json ]; then local why; why=$(grep -aE "Engine core|CUDA|Error|error" logs/run_vllm_b${B}_$ARM.log | tail -2 | tr '\n' ' ' | cut -c1-400)
    python -c "import json,sys; json.dump({'engine':'vllm','arm':'$ARM','batch':$B,'status':('alarm' if $rc==142 else 'vllm_init_failed'),'rc':$rc,'reason':sys.argv[1]}, open('$W/vllm_b${B}_$ARM.json','w'))" "$why"; fi
  grep -aE "P37VLLM|Marlin|MARLIN|Capturing CUDA graphs|Actual usage|Available KV cache|Engine core" logs/run_vllm_b${B}_$ARM.log | grep -v "^\s*$" | tail -4 | cut -c1-240 | sed "s/^/    /"
  { echo -n "vllm/$ARM B=$B rc=$rc "; grep -a "P37VLLM" logs/run_vllm_b${B}_$ARM.log | tail -1 | cut -c1-300; echo; } >> summary.txt; }
cd $W || exit 1; [ -f TP_DONE ] || { echo "p37b: lane not finished (no TP_DONE)"; exit 9; }
# Guard (16:30Z): the lane's own resume recovered the GPTQ fetch at 16:25:30Z ("staged in 19.7 min") and the vLLM arms are running
# in-lane; p37b then has nothing to do. It acts only if the lane wrote SKIPPED (no comparator) rows.
if ! grep -q "SKIPPED (no comparator)" summary.txt 2>/dev/null; then echo "p37b: NOT NEEDED -- the comparator ran in-lane (no SKIPPED rows); nothing re-run" | tee -a summary.txt; touch TP2_DONE; exit 0; fi
VLLM_VER=$($W/venv-vllm/bin/python -c "import vllm; print(vllm.__version__)" 2>/dev/null); [ -n "$VLLM_VER" ] || { echo "p37b: vllm venv unusable"; touch TP2_DONE; exit 9; }
PSHA1=$(python -c "import json; print(json.load(open('/root/p37/prompts_b1.json'))['prompts_sha256'])"); PSHA16=$(python -c "import json; print(json.load(open('/root/p37/prompts_b16.json'))['prompts_sha256'])")
echo "AMENDMENT 2 (p37b): GPTQ fetch resumed after the lane's alarm; vLLM arms follow on the same prompts (sha B1=$PSHA1 B16=$PSHA16)" | tee -a summary.txt
VLLM_OK=0; for a in 1 2 3; do say "gptq fetch attempt $a (resume)"; fetch $GPTQ_MID $GPTQ_REV gptq 2400 && { VLLM_OK=1; break; }; sleep 30; done
[ "$VLLM_OK" = "1" ] || { echo "p37b: GPTQ NEVER STAGED after 3 attempts -- NO COMPARATOR" | tee -a summary.txt; touch TP2_DONE; exit 11; }
say "p37b: vLLM arms, registered order"
for B in 1 16; do vllm_arm graph_r1 $B; vllm_arm eager $B; vllm_arm fp8kv $B; done
for B in 1 16; do vllm_arm graph_r2 $B; done
say "reduce (p37b)"; python $W/p37_reduce.py $W --md $W/RESULTS-p37.md | tee RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt
say "P37B DONE"; touch TP2_DONE
