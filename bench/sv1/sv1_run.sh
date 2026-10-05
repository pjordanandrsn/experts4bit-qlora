#!/bin/bash
# bench/sv1/sv1_run.sh -- lane SV1, BOX side (bench/sv1/SV1-PREREG.md). Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER
# (TC1_EXTRA_STAGE carries this file, sv1_measure.py and bench/p39/calib.json), so it speaks tc1_drive's contract: the nonce
# handshake (TC1_RUN_NONCE within 30 s), summary.txt one line per finished arm, TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> /
# TP_DONE.<nonce> at the end -- a refusal writes them too. OLMoE-1B-7B first (eager, decode graphs, decode graphs + prefill
# graph), each from an arena baked here; then Qwen3-30B-A3B (decode graphs, + prefill graph) only if the OLMoE eager arm is OK.
# Nothing here creates, destroys or approves compute.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] sv1: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts arenas
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
OLMOE=allenai/OLMoE-1B-7B-0924; OLMOE_REV=6d84c48581ece794365f2b8e9cfb043c68ade9c5
QWEN=Qwen/Qwen3-30B-A3B; QWEN_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
[ -s calib.json ] || { echo "CALIB MISSING" | tee -a summary.txt; finish 9; }

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
from experts4bit_qlora.serve_recipe import ServeSetup, estimate_serve_footprint, min_hot_rows   # the API this lane measures
from experts4bit_qlora.engines.fp8_paged_kv import fused_append_unsupported
from nvme_bake_nf4 import bake_nf4  # noqa: F401
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    got = json.loads(md.distribution(dist).read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability()
assert fused_append_unsupported(cap) is None, f"decode graphs cannot run on sm_{cap[0]}{cap[1]}: this lane needs sm_89+"
print("sv1 tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name(), cap)
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt

fetch(){ perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$1', revision='$2', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja']))" 2> "logs/fetch_$3.log" | tail -1; }
bake(){ local snap=$1 out=$2 tag=$3
  say "bake $tag -> $out"
  perl -e 'alarm 3600; exec @ARGV' python -c "from nvme_bake_nf4 import bake_nf4; bake_nf4('$snap', '$out')" > "logs/bake_$tag.log" 2>&1
  local rc=$?; echo "bake $tag rc=$rc $(du -sh "$out" 2>/dev/null | cut -f1) $(tail -1 "logs/bake_$tag.log" | cut -c1-160)" | tee -a summary.txt; return $rc; }
arm(){ local tag=$1 model=$2 rev=$3 arena=$4 graphs=$5 pg=$6
  say "arm $tag (graphs $graphs, prefill graph $pg)"
  perl -e 'alarm 3600; exec @ARGV' python sv1_measure.py --model "$model" --revision "$rev" --arena "$arena" --calib calib.json \
      --graphs "$graphs" --prefill-graph "$pg" --out "receipts/$tag.json" > "logs/arm_$tag.log" 2>&1
  local rc=$?; echo "$tag rc=$rc $(tail -1 "logs/arm_$tag.log" | cut -c1-240)" | tee -a summary.txt; return $rc; }

snap=$(fetch $OLMOE $OLMOE_REV olmoe); [ -d "$snap" ] || { echo "olmoe: FETCH FAILED" | tee -a summary.txt; finish 10; }
echo "olmoe: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
bake "$snap" arenas/olmoe.nf4 olmoe || finish 13
# The eager anchor doubles as the instrument's own smoke test: if it is not OK, stop here, before the 57 GB fetch.
arm olmoe_eager $OLMOE $OLMOE_REV arenas/olmoe.nf4 0 0 || { tail -20 logs/arm_olmoe_eager.log; echo "ANCHOR ARM FAILED: stopping before the Qwen3 fetch" | tee -a summary.txt; finish 12; }
arm olmoe_graphs $OLMOE $OLMOE_REV arenas/olmoe.nf4 1 0
arm olmoe_prefill $OLMOE $OLMOE_REV arenas/olmoe.nf4 1 1
rm -rf "$(dirname "$(dirname "$snap")")" arenas/olmoe.nf4*
snap=$(fetch $QWEN $QWEN_REV qwen3); [ -d "$snap" ] || { echo "qwen3: FETCH FAILED" | tee -a summary.txt; finish 10; }
echo "qwen3: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
bake "$snap" arenas/qwen3.nf4 qwen3 || finish 13
arm qwen3_graphs $QWEN $QWEN_REV arenas/qwen3.nf4 1 0
arm qwen3_prefill $QWEN $QWEN_REV arenas/qwen3.nf4 1 1
n=$(ls receipts/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "SV1 done: $n arm receipts" | tee -a summary.txt
[ "$n" = 5 ] && finish 0 || finish 11
