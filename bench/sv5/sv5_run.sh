#!/bin/bash
# bench/sv5/sv5_run.sh -- lane SV5, BOX side (bench/sv5/SV5-PREREG.md). Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER
# (TC1_EXTRA_STAGE carries this file, bench/sv4/sv4_measure.py and bench/p39/calib.json), so it speaks tc1_drive's contract: the nonce
# handshake (TC1_RUN_NONCE within 30 s), summary.txt one line per finished arm, TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> /
# TP_DONE.<nonce> at the end -- a refusal writes them too. Qwen3-30B-A3B on an RTX 4090, the planner's all-VRAM plan for
# 8 x 8192 after SV4: one arena baked here, then a5_short (1,024-token prompts, the anchor) and a5_long (8,000-token prompts).
# Nothing here creates, destroys or approves compute.
# Exit codes (SV5-PREREG "Outcomes"). 13 and 18 keep their host meanings, the only ones the launcher reads as evidence
# against the machine: 13 = under the disk floor, 18 = under the host-RAM floor. Everything else is the workload's:
# 9 install / tripwire, 10 fetch, 11 fewer than two receipts, 12 the bake, 16 the a5_short anchor not finishing.
# sv5-4090-1 ran this file before that split, when a bake failure also exited 13; it exited 0.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] sv5: $*"; }
# Every large file lives OUTSIDE $W, whose contents tc1_drive's final rsync copies back whole (sv1-5090-1's lesson, #1160):
# the arena under /root/sv5-arena, the checkpoint in the default HF cache (/root/.cache).
ARENAS=/root/sv5-arena
finish(){ local rc=$1; rm -rf "$ARENAS"; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts "$ARENAS"
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
QWEN=Qwen/Qwen3-30B-A3B; QWEN_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
[ -s calib.json ] || { echo "CALIB MISSING" | tee -a summary.txt; finish 9; }
FREE_GB=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 120 ] || { echo "BOX_REFUSED disk=${FREE_GB:-?}GB < 120 GB (57 GB checkpoint + 16 GB arena + venv)" | tee -a summary.txt; finish 13; }
# A guard, not a measured need: the bake and the load stream the checkpoint shard by shard, and sv5-4090-1's arms
# peaked at 2.4 GB of anonymous host memory.
MIN_RAM_GB=32
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
CG=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max); CG_GB=$([ "$CG" = max ] && echo 99999 || echo $(( CG / 1073741824 )))
[ "${AVAIL_GB:-0}" -ge "$MIN_RAM_GB" ] && [ "$CG_GB" -ge "$MIN_RAM_GB" ] || { echo "BOX_REFUSED ram: available ${AVAIL_GB:-?} GB, cgroup ${CG_GB} GB < $MIN_RAM_GB GB" | tee -a summary.txt; finish 18; }

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
from experts4bit_qlora.serve_recipe import ServeSetup, estimate_serve_footprint, dram_rows_per_layer  # noqa: F401
from experts4bit_qlora.engines.host_heap import release_freed_host_heap
from experts4bit_qlora.engines.fp8_paged_kv import fused_append_unsupported
from nvme_bake_nf4 import bake_nf4  # noqa: F401
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    got = json.loads(md.distribution(dist).read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability()
assert fused_append_unsupported(cap) is None, f"decode graphs cannot run on sm_{cap[0]}{cap[1]}: this lane needs sm_89+"
print("sv5 tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name(), cap,
      "malloc_trim:", release_freed_host_heap())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt

snap=$(perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$QWEN', revision='$QWEN_REV', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja']))" 2> logs/fetch_qwen3.log | tail -1)
[ -d "$snap" ] || { echo "qwen3: FETCH FAILED" | tee -a summary.txt; finish 10; }
echo "qwen3: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
say "bake qwen3"
perl -e 'alarm 3600; exec @ARGV' python -c "from nvme_bake_nf4 import bake_nf4; bake_nf4('$snap', '$ARENAS/qwen3.nf4')" > logs/bake_qwen3.log 2>&1
rc=$?; echo "bake qwen3 rc=$rc $(du -sh "$ARENAS/qwen3.nf4" 2>/dev/null | cut -f1) $(tail -1 logs/bake_qwen3.log | cut -c1-160)" | tee -a summary.txt
[ $rc = 0 ] || { tail -5 logs/bake_qwen3.log; echo "BAKE FAIL" | tee -a summary.txt; finish 12; }
arm(){ local tag=$1 prompt=$2
  say "arm $tag (all-VRAM, 8 x 8192, decode graphs, buckets 1,2,4,8, $prompt-token prompts)"
  perl -e 'alarm 3600; exec @ARGV' python sv4_measure.py --model $QWEN --revision $QWEN_REV --arena "$ARENAS/qwen3.nf4" --calib calib.json \
      --placement all-vram --max-seqs 8 --context 8192 --graphs 1 --prefill-graph 0 --buckets 1,2,4,8 --prompt-tokens "$prompt" \
      --new-tokens 32 --out "receipts/$tag.json" > "logs/arm_$tag.log" 2>&1
  local rc=$?; echo "$tag rc=$rc $(tail -1 "logs/arm_$tag.log" | cut -c1-320)" | tee -a summary.txt; return $rc; }
# The short-prompt arm doubles as the instrument's smoke test: if it is not OK, stop before the long one.
arm a5_short 1024 || { tail -20 logs/arm_a5_short.log; echo "ANCHOR ARM FAILED: stopping before the long-prompt arm" | tee -a summary.txt; finish 16; }
arm a5_long 8000
n=$(ls receipts/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "SV5 done: $n arm receipts" | tee -a summary.txt
[ "$n" = 2 ] && finish 0 || finish 11
