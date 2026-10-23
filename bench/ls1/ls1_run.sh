#!/bin/bash
# bench/ls1/ls1_run.sh -- LS1, BOX side (an EXPLORE screen, never a reading; Map 1 T-A, e4b#945). Started by
# bench/tc1/tc1_drive.sh as its TC1_RUNNER (TC1_EXTRA_STAGE carries this file, ls1_oracle.py and the ladder), so it speaks
# tc1_drive's contract: the nonce handshake, summary.txt, and TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> / TP_DONE.<nonce>
# at the end, a refusal included. Nothing here creates, destroys or approves compute.
#   install gnf4 + e4b at the pins -> tripwire -> fetch Qwen3-30B-A3B at TC1's pin -> ls1_oracle.py's self-test -> the screen
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] ls1: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
echo "EXPLORE (LS1 screen, Map 1 T-A): not a reading" | tee -a summary.txt
for f in ls1_oracle.py ladder.py; do [ -s $W/$f ] || { echo "STAGE MISSING: $f" | tee -a summary.txt; finish 9; }; done
mkdir -p $W/moegen && cp $W/ladder.py $W/moegen/ladder.py && mkdir -p $W/ls1 && cp $W/ls1_oracle.py $W/ls1/ls1_oracle.py

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
from experts4bit_qlora.engines import fast
assert callable(fast._group_by_expert), "e4b without the single-read grouping"
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    d = json.loads(md.distribution(dist).read_text("direct_url.json"))
    got = d.get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
print("ls1 tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt
python ls1/ls1_oracle.py --self-test 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { echo "SELF-TEST FAIL" | tee -a summary.txt; finish 21; }

say "fetch Qwen/Qwen3-30B-A3B@ad44e777 (TC1's pin)"
snap=$(perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-30B-A3B', revision='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39',
                        allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.jinja']))" 2> logs/fetch.log | tail -1)
[ -d "$snap" ] || { tail -3 logs/fetch.log; echo "FETCH FAILED" | tee -a summary.txt; finish 22; }
echo "fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt

say "screen: fused, oracle_rc, oracle, legacy13 (mirrored)"
perl -e "alarm ${LS1_ALARM_S:-3600}; exec @ARGV" python ls1/ls1_oracle.py --model "$snap" --steps ${LS1_STEPS:-10} --warmup 3 \
  --seq 512 --mb 1 --out receipts/ls1_oracle.json > logs/ls1_oracle.log 2>&1
rc=$?
grep -E "SUMMARY|patched|rc shape" logs/ls1_oracle.log | tee -a summary.txt
echo "ls1_oracle rc=$rc" | tee -a summary.txt
finish $rc
