#!/bin/bash
# bench/th/th_run.sh -- the T-H census, BOX side (EXPLORE, never a reading; Map 1, bench/train-capture/DESIGN-TB.md section 4).
# Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER (TC1_EXTRA_STAGE carries this file, th_census.py and the ladder), so it
# speaks tc1_drive's contract: the nonce handshake, summary.txt, and TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> /
# TP_DONE.<nonce> at the end, a refusal included. Nothing here creates, destroys or approves compute.
#   install gnf4 + e4b at the pins -> tripwire -> fetch Qwen3-30B-A3B at TC1's pin -> th_census.py's self-test -> the census
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] th: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
echo "EXPLORE (T-H census, Map 1 T-B): not a reading" | tee -a summary.txt
for f in th_census.py ladder.py; do [ -s $W/$f ] || { echo "STAGE MISSING: $f" | tee -a summary.txt; finish 9; }; done
mkdir -p $W/moegen $W/th && cp $W/ladder.py $W/moegen/ladder.py && cp $W/th_census.py $W/th/th_census.py

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA, transformers 5.18.0"
perl -e 'alarm 2400; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.18.0" accelerate safetensors "huggingface_hub>=0.23" datasets sentencepiece > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, os, torch
from torch.profiler import profile
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    d = json.loads(md.distribution(dist).read_text("direct_url.json"))
    got = d.get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
print("th tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt
python th/th_census.py --self-test 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { echo "SELF-TEST FAIL" | tee -a summary.txt; finish 21; }

say "fetch Qwen/Qwen3-30B-A3B@ad44e777 (TC1's pin)"
snap=$(perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-30B-A3B', revision='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39',
                        allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.jinja']))" 2> logs/fetch.log | tail -1)
[ -d "$snap" ] || { tail -3 logs/fetch.log; echo "FETCH FAILED" | tee -a summary.txt; finish 22; }
echo "fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt

say "census: one profiled optimizer step (seq 512, batch 1, 4 micro-batches, r 16, fp32 adapters)"
perl -e "alarm ${TH_ALARM_S:-3600}; exec @ARGV" python th/th_census.py --model "$snap" --seq 512 --accum 4 --r 16 --alpha 16 \
  --adapter-dtype fp32 --out receipts/th_census.json > logs/th_census.log 2>&1
rc=$?
grep -E "^\[th\]" logs/th_census.log | tee -a summary.txt
echo "th_census rc=$rc" | tee -a summary.txt
finish $rc
