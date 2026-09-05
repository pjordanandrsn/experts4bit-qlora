#!/bin/bash
# Lane tp1 (training parity, all six serving families) -- pre-registered in P36-PREREG.md.
# Cut = the shipped code: e4b main @f4b639fd2640 (= v0.35.0 + #396 docs) + gnf4 main @ddcb850e05c3 (= v0.30.0 + #341 docs).
# Pattern: bo5_run.sh (per-family functions, per-arm `perl -e 'alarm N'`, summary.txt, TP_DONE, `say` timestamps).
# Arms per family: reference / fused (enable_fast_train dgrad=True) / batched (enable_batched_train); gpt-oss: attn_only (+ refusal
# probes) and the EXPERIMENTAL gnf4 MXFP4 runner; Mixtral on offload=1. One JSON per arm: <fam>_train_<arm>.json. No serving arm.
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
: > summary.txt
# ---------------------------------------------------------------- install + tripwire (the cut, by symbol AND version)
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken > logs/pip.log 2>&1
rc=$?; echo "pip rc=$rc"; [ $rc -ne 0 ] && { tail -4 logs/pip.log; echo "PIP FAIL"; touch TP_DONE; exit 9; }
python - <<'PYT' || { echo "TRIPWIRE FAIL"; touch TP_DONE; exit 9; }
import importlib.metadata as md, inspect
import experts4bit_qlora as e
from experts4bit_qlora import enable_fast_train, enable_batched_train, ExpertsLoRA, load_moe_4bit_streaming, verify_moe_4bit
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated   # the 0.35.0 symbol bo7 asserts too
from experts4bit_qlora.engines.batched import _dequant_whole, _PAD_WASTE_LIMIT
from nf4_qlora import fused_grouped_lora
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters, "nf4_qlora.fused_grouped_lora has no dgrad_kernel"
from mxfp4_qlora import ExpertsMxfp4LoRA
import run_mxfp4_20b_qlora
assert e.__version__ == "0.35.0", e.__version__
assert md.version("grouped-nf4-gemm") == "0.30.0", md.version("grouped-nf4-gemm")
print("tp1 tripwire OK: e4b", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "(no usercustomize hook: training lane)")
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt
grep MemTotal /proc/meminfo | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX"; touch TP_DONE; exit 10; }
[ -s /root/.cache/huggingface/token ] || { echo "TOKEN MISSING (gemma-4-it and Mixtral-Instruct are gated)"; touch TP_DONE; exit 8; }
df -h /root | tail -1
# ---------------------------------------------------------------- repo helpers at the cut (self-contained: no private helper is used)
say "fetching repo helpers at e4b @$E4B_SHA"
rm -rf $W/e4b-src; mkdir -p $W/e4b-src && perl -e 'alarm 600; exec @ARGV' curl -sSL --retry 3 -o $W/e4b-src.tgz "https://github.com/pjordanandrsn/experts4bit-qlora/archive/$E4B_SHA.tar.gz" && tar -xzf $W/e4b-src.tgz -C $W/e4b-src --strip-components=1; rc=$?; cd $W   # P36 amendment 2: archive tarball instead of git fetch-by-sha (the fetch failed on the box)
[ $rc -ne 0 ] && { echo "SRC FETCH FAIL"; touch TP_DONE; exit 9; }
cp $W/e4b-src/bench/flagship-matrix/drivers/n9_datasets.py $W/e4b-src/bench/flagship-matrix/ds_manifest.json \
   $W/e4b-src/bench/train-anchor/train_anchor.py $W/e4b-src/bench/train-anchor/train_anchor_gate.py $W/
# ---------------------------------------------------------------- box class (bench/train-anchor): a training receipt without its class is not a receipt
say "train anchor"
ANCHOR_OUT=$W/anchor.json perl -e 'alarm 900; exec @ARGV' python $W/train_anchor.py > logs/anchor.log 2>&1; tail -3 logs/anchor.log
python $W/train_anchor_gate.py $W/anchor.json | tee logs/anchor_gate.log; arc=${PIPESTATUS[0]}
export TP1_ANCHOR_JSON=$W/anchor.json TP1_BOX_CLASS="$(grep -E '^\s*class ' logs/anchor_gate.log | awk '{print $2}')"
echo "ANCHOR rc=$arc class=$TP1_BOX_CLASS" | tee -a summary.txt
if [ "$arc" -ne 0 ] && [ "$ANCHOR_STRICT" = "1" ]; then echo "BOX REFUSED by train anchor (rc=$arc)"; touch BOX_REFUSED TP_DONE; exit 12; fi
# ---------------------------------------------------------------- the fixed text: regenerate, then REFUSE unless the bytes are the registered ones
say "dataset: $DATASET (n9_datasets.py, sha-verified against ds_manifest.json)"
mkdir -p $W/data && (cd $W/data && python $W/n9_datasets.py $W/data > $W/logs/datasets.log 2>&1); tail -2 logs/datasets.log
DATA=$W/data/ds_$DATASET.json
DATA_SHA=$(python -c "import json,sys; print(json.load(open('$W/ds_manifest.json'))['$DATASET']['sha256'])")
GOT_SHA=$(sha256sum $DATA | awk '{print $1}')
[ "$GOT_SHA" = "$DATA_SHA" ] || { echo "DATASET MISMATCH: $GOT_SHA != $DATA_SHA"; touch TP_DONE; exit 13; }
echo "DATASET $DATASET sha=$DATA_SHA" | tee -a summary.txt
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
skip granite || { fetch $GR granite 2400 && { arm granite reference 1200 $GR 0; arm granite fused 1200 $GR 0; arm granite batched 1200 $GR 0; echo "GRANITE DONE"; }
  free_family granite ibm-granite--granite-3.1-3b-a800m-instruct; }
# OLMoE-1B-7B-0924-Instruct: first registered-text parity (dgrad-gate used synthetic tokens); the serving checkpoint (resident)
OL=allenai/OLMoE-1B-7B-0924-Instruct
skip olmoe || { fetch $OL olmoe 2400 && { arm olmoe reference 1500 $OL 0; arm olmoe fused 1500 $OL 0; arm olmoe batched 1500 $OL 0; echo "OLMOE DONE"; }
  free_family olmoe allenai--OLMoE-1B-7B-0924-Instruct; }
# gpt-oss-20b: attention-only QLoRA over the BARE experts (+ refusal probes -> fused/batched stubs), then the EXPERIMENTAL gnf4 MXFP4 runner
GO=openai/gpt-oss-20b
skip gptoss || { fetch $GO gptoss 3000 && { arm gptoss attn_only 2400 $GO 0 --probe-refusals 1
    say "arm gptoss/mxfp4 (EXPERIMENTAL: gnf4 run_mxfp4_20b_qlora, its own wikitext-2 text; never licensed by this lane)"
    perl -e 'alarm 5400; exec @ARGV' python -u -m run_mxfp4_20b_qlora --model $GO --out $W/gptoss_mxfp4 --steps $STEPS --seq $SEQ \
        --r 8 --alpha 16 --lr 1e-4 --seed 0 > logs/run_gptoss_mxfp4.log 2>&1; rc=$?
    python - "$rc" <<'PYM'
import json, os, sys
rc = int(sys.argv[1]); W = "/root/tp1"; art = os.path.join(W, "gptoss_mxfp4", "run_artifact.json")
rec = {"fam": "gptoss", "arm": "mxfp4", "experimental": True, "licensed": False, "rc": rc,
       "route": "grouped-nf4-gemm mxfp4_qlora.ExpertsMxfp4LoRA via run_mxfp4_20b_qlora (own text: wikitext-2 train)"}
if os.path.exists(art):
    rec["status"] = "ok" if rc == 0 else "artifact_with_nonzero_rc"; rec["artifact"] = json.load(open(art))
    steps = [json.loads(l) for l in open(os.path.join(W, "gptoss_mxfp4", "steps.jsonl"))]
    rec["losses"] = [s["loss"] for s in steps]; rec["step_s"] = [s["dt"] for s in steps]
    rec["peak_vram_gb"] = max(s["cuda_peak_gb"] for s in steps) if steps else None
else:
    rec["status"] = "alarm" if rc == 142 else "failed"; rec["reason"] = f"rc={rc}, no run_artifact.json"
json.dump(rec, open(os.path.join(W, "gptoss_train_mxfp4.json"), "w"), indent=1)
print("CELL", rec["status"].upper(), "gptoss/mxfp4 EXPERIMENTAL", json.dumps(rec.get("artifact", {}).get("canary", {})))
PYM
    { echo -n "gptoss/mxfp4 rc=$rc EXPERIMENTAL "; grep -aE "canary|eval@|PROVENANCE" logs/run_gptoss_mxfp4.log | tail -3 | tr '\n' ' ' | cut -c1-400; echo; } >> summary.txt
    echo "GPTOSS DONE"; }
  free_family gptoss openai--gpt-oss-20b; }
# Qwen3-30B-A3B: the family with the registered receipts (resident on 32 GB; dgrad-gate measured 23-26 GB peak resident)
Q=Qwen/Qwen3-30B-A3B
skip qwen3 || { fetch $Q qwen3 4800 && { arm qwen3 reference 3000 $Q 0; arm qwen3 fused 2400 $Q 0; arm qwen3 batched 3000 $Q 0; echo "QWEN3 DONE"; }
  free_family qwen3 Qwen--Qwen3-30B-A3B; }
# Gemma-4-26B-A4B-it: #344 risk -- a CUDA load fault is a row (exit 6), not a retry
GM=google/gemma-4-26B-A4B-it
skip gemma4 || { fetch $GM gemma4 4800 && { arm gemma4 reference 3000 $GM 0; arm gemma4 fused 2400 $GM 0; arm gemma4 batched 3000 $GM 0; echo "GEMMA4 DONE"; }
  free_family gemma4 google--gemma-4-26B-A4B-it; }
# Mixtral-8x7B-Instruct-v0.1: 25.4 GB of NF4 experts -> offload=1 (pinned host RAM); the resident probe is opt-in and expected to OOM
MX=mistralai/Mixtral-8x7B-Instruct-v0.1
skip mixtral || { fetch $MX mixtral 7200 && { arm mixtral reference 4200 $MX 1; arm mixtral fused 3600 $MX 1; arm mixtral batched 4200 $MX 1
    # opt-in resident probe: the reference arm, 3 steps, resident -- expected to OOM (P5); its JSON is mixtral_train_resident_probe.json
    [ "$MIXTRAL_RESIDENT_PROBE" = "1" ] && ARMOPT=reference STEPS=3 arm mixtral resident_probe 1800 $MX 0
    echo "MIXTRAL DONE"; }
  free_family mixtral mistralai--Mixtral-8x7B-Instruct-v0.1; }
# ---------------------------------------------------------------- reduce, summarise, mark
say "reduce"
python $W/tp1_reduce.py $W --md $W/RESULTS-tp1.md | tee RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt
say "TP_DONE"; touch TP_DONE
