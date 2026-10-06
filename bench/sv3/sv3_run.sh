#!/bin/bash
# bench/sv3/sv3_run.sh -- lane SV3, BOX side (bench/sv3/SV3-PREREG.md). Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER
# (TC1_EXTRA_STAGE carries this file, sv3_measure.py, bench/p98/p98_bake.py and bench/p39/calib.json), so it speaks
# tc1_drive's contract: the nonce handshake (TC1_RUN_NONCE within 30 s), summary.txt one line per finished arm,
# TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at the end -- a refusal writes them too.
# Qwen3.6-35B-A3B first (the eager arm is the anchor), then gpt-oss-20b; each model's arena is baked here through
# experts4bit-qlora's loader (p98_bake.py) and freed before the next. Nothing here creates, destroys or approves compute.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] sv3: $*"; }
# Every large file lives OUTSIDE $W, whose contents tc1_drive's final rsync copies back whole (#1160): the bakes' work
# dirs under /root/sv3-work-*, the checkpoints in the default HF cache (/root/.cache).
finish(){ local rc=$1; rm -rf /root/sv3-work-*; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
QWEN=Qwen/Qwen3.6-35B-A3B; QWEN_REV=995ad96eacd98c81ed38be0c5b274b04031597b0
GPTOSS=openai/gpt-oss-20b; GPTOSS_REV=6cee5e81ee83917806bbde320786a8fb61efebee
[ -s calib.json ] || { echo "CALIB MISSING" | tee -a summary.txt; finish 9; }
FREE_GB=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 150 ] || { echo "BOX_REFUSED disk=${FREE_GB:-?}GB < 150 GB (72 GB checkpoint + 17 GB NF4 snapshot + 17 GB arena + venv)" | tee -a summary.txt; finish 13; }

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA, transformers 5.18.0"
perl -e 'alarm 2400; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.18.0" "bitsandbytes==0.50.2" accelerate safetensors "huggingface_hub>=0.23" sentencepiece fastapi > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, os, torch
from experts4bit_qlora.serve_recipe import ServeSetup, estimate_serve_footprint, linear_state_pool_bytes  # noqa: F401
from experts4bit_qlora.engines.linear_state import state_geometry  # noqa: F401
from experts4bit_qlora.engines.fp8_paged_kv import fused_append_unsupported
from nvme_arena import bake_expert_tensors  # noqa: F401
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    got = json.loads(md.distribution(dist).read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
from transformers.cache_utils import LinearAttentionLayer  # noqa: F401  -- the per-slot state carrier (transformers >= 5.13)
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability()
assert fused_append_unsupported(cap) is None, f"decode graphs cannot run on sm_{cap[0]}{cap[1]}: this lane needs sm_89+"
print("sv3 tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name(), cap)
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt

fetch(){ perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$1', revision='$2', ignore_patterns=['original/*', 'metal/*', 'consolidated*'], allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja']))" 2> "logs/fetch_$3.log" | tail -1; }
bake(){ local model=$1 rev=$2 tag=$3
  say "bake $tag"
  perl -e 'alarm 3600; exec @ARGV' python p98_bake.py --model "$model" --revision "$rev" --work "/root/sv3-work-$tag" > "logs/bake_$tag.log" 2>&1
  local rc=$?; echo "bake $tag rc=$rc $(python -c "import json; r=json.load(open('/root/sv3-work-$tag/bake.json')); print({k: r.get(k) for k in ('status','layers','experts','load_s','bake_s','err')})" 2>&1 | cut -c1-200)" | tee -a summary.txt
  [ -s "/root/sv3-work-$tag/nf4.arena" ] && return $rc || return 13; }
arm(){ local tag=$1 model=$2 rev=$3 arena=$4 seqs=$5 graphs=$6 buckets=$7
  say "arm $tag (seqs $seqs, graphs $graphs, buckets $buckets)"
  perl -e 'alarm 3600; exec @ARGV' python sv3_measure.py --model "$model" --revision "$rev" --arena "$arena" --calib calib.json \
      --graphs "$graphs" --prefill-graph 0 --buckets "$buckets" --max-seqs "$seqs" --out "receipts/$tag.json" > "logs/arm_$tag.log" 2>&1
  local rc=$?; echo "$tag rc=$rc $(tail -1 "logs/arm_$tag.log" | cut -c1-320)" | tee -a summary.txt; return $rc; }

snap=$(fetch $QWEN $QWEN_REV qwen36); [ -d "$snap" ] || { echo "qwen36: FETCH FAILED" | tee -a summary.txt; finish 10; }
echo "qwen36: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
bake $QWEN $QWEN_REV qwen36 || finish 13
QA=/root/sv3-work-qwen36/nf4.arena
# The eager arm doubles as the instrument's smoke test: if it is not OK, stop before the other arms and the gpt-oss fetch.
arm q36_e16 $QWEN $QWEN_REV $QA 16 0 1,2,4,8,16 || { tail -20 logs/arm_q36_e16.log; echo "ANCHOR ARM FAILED: stopping" | tee -a summary.txt; finish 12; }
arm q36_g16 $QWEN $QWEN_REV $QA 16 1 1,2,4,8,16
arm q36_g1_default $QWEN $QWEN_REV $QA 1 1 1,2,4,8,16
arm q36_g1_capped $QWEN $QWEN_REV $QA 1 1 1
rm -rf /root/sv3-work-qwen36 "$(dirname "$(dirname "$snap")")"
say "freed qwen36 ($(df -h /root | tail -1 | awk '{print $4}') free)"
snap=$(fetch $GPTOSS $GPTOSS_REV gptoss); [ -d "$snap" ] || { echo "gptoss: FETCH FAILED" | tee -a summary.txt; finish 10; }
echo "gptoss: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
if bake $GPTOSS $GPTOSS_REV gptoss; then
  arm gptoss_g16 $GPTOSS $GPTOSS_REV /root/sv3-work-gptoss/nf4.arena 16 1 1,2,4,8,16
fi
n=$(ls receipts/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "SV3 done: $n arm receipts" | tee -a summary.txt
[ "$n" = 5 ] && finish 0 || finish 11
