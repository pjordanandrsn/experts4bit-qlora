#!/bin/bash
# bench/fp1/fp1_run.sh -- lane FP1, BOX side (bench/fp1/FP1-PREREG.md). Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER
# (TC1_EXTRA_STAGE carries this file and fp1_measure.py), so it speaks tc1_drive's contract: the nonce handshake (TC1_RUN_NONCE
# within 30 s), summary.txt one line per finished arm, TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at the end --
# a refusal writes them too. Three arms, one process each, in order: OLMoE-1B-7B resident (the anchor read on an RTX A2000
# before this lane), Qwen3-30B-A3B resident, Qwen3-30B-A3B with host-resident (pinned) experts. Registered shape: seq 512 x
# micro-batch 2, 12 steps, r 8 / alpha 16 bf16 adapters, the grouped NF4 kernel, AdamW 2e-4. Nothing here creates, destroys
# or approves compute.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] fp1: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
OLMOE=allenai/OLMoE-1B-7B-0924; OLMOE_REV=6d84c48581ece794365f2b8e9cfb043c68ade9c5
QWEN=Qwen/Qwen3-30B-A3B; QWEN_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA, transformers 5.18.0"
perl -e 'alarm 2400; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.18.0" "bitsandbytes==0.50.2" datasets accelerate safetensors "huggingface_hub>=0.23" sentencepiece > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, os, torch
from experts4bit_qlora import describe_moe, estimate_qlora_footprint, prepare_qlora_training   # the API this lane measures
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    got = json.loads(md.distribution(dist).read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
print("fp1 tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt

fetch(){ perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$1', revision='$2', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja']))" 2> "logs/fetch_$3.log" | tail -1; }
arm(){ local tag=$1 model=$2 rev=$3 res=$4
  say "arm $tag ($res)"
  perl -e 'alarm 5400; exec @ARGV' python fp1_measure.py --model "$model" --revision "$rev" --residency "$res" --out "receipts/$tag.json" > "logs/arm_$tag.log" 2>&1
  local rc=$?; echo "$tag rc=$rc $(tail -1 "logs/arm_$tag.log" | cut -c1-200)" | tee -a summary.txt; }

snap=$(fetch $OLMOE $OLMOE_REV olmoe); [ -d "$snap" ] || { echo "olmoe: FETCH FAILED" | tee -a summary.txt; finish 10; }
echo "olmoe: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
arm olmoe_device $OLMOE $OLMOE_REV device
rm -rf "$(dirname "$(dirname "$snap")")"
snap=$(fetch $QWEN $QWEN_REV qwen3); [ -d "$snap" ] || { echo "qwen3: FETCH FAILED" | tee -a summary.txt; finish 10; }
echo "qwen3: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
arm qwen3_device $QWEN $QWEN_REV device
arm qwen3_host $QWEN $QWEN_REV host
n=$(ls receipts/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "FP1 done: $n arm receipts" | tee -a summary.txt
[ "$n" = 3 ] && finish 0 || finish 11
