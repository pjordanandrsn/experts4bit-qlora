#!/bin/bash
# bench/det/det_run.sh -- the training path's deterministic mode, base-only self-pair, BOX side (EXPLORE, never a reading).
# Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER (TC1_EXTRA_STAGE carries this file, det_selfpair.py, the ladder and
# tests/test_det_gather.py with its tests/quant_guard.py), so it speaks tc1_drive's contract: the nonce handshake, summary.txt, and
# TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at the end, a refusal included. Nothing here creates,
# destroys or approves compute.
#   install e4b (with E4B_TRAIN_DET_GATHER, e4b#2054) + gnf4 -> tripwire -> test_det_gather.py ON THE CARD (its CUDA
#   cases are (b): the gather's backward bitwise equal to itself) and the record-only gather probe -> fetch Qwen3-30B-A3B
#   -> det_selfpair.py's self-test -> one process per arm: A, B, C, Cp
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] det: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
echo "EXPLORE (the training path's deterministic mode, base-only self-pair): not a reading" | tee -a summary.txt
for f in det_selfpair.py ladder.py test_det_gather.py quant_guard.py; do [ -s $W/$f ] || { echo "STAGE MISSING: $f" | tee -a summary.txt; finish 9; }; done
mkdir -p $W/moegen $W/det $W/cardtests && cp $W/ladder.py $W/moegen/ladder.py && cp $W/det_selfpair.py $W/det/ && cp $W/test_det_gather.py $W/quant_guard.py $W/cardtests/

free_gb=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 120 ] || { echo "FLOOR: $free_gb GB free < 120" | tee -a summary.txt; finish 13; }
[ "${mem_mib:-0}" -ge 30000 ] || { echo "FLOOR: $mem_mib MiB card < 30000" | tee -a summary.txt; finish 13; }

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA, transformers 5.18.0"
perl -e 'alarm 2400; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.18.0" accelerate safetensors "huggingface_hub>=0.23" datasets sentencepiece "brotli>=1.2.0" pytest > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, os, torch
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    d = json.loads(md.distribution(dist).read_text("direct_url.json"))
    got = d.get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
from experts4bit_qlora.engines import fast
assert hasattr(fast, "det_gather_enabled") and "det_gather" in fast.FAST_TRAIN_STATS   # e4b#2054 is installed
import torch.utils.deterministic
print("det tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__,
      torch.cuda.get_device_name(), "sm_%d%d" % torch.cuda.get_device_capability())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt

say "premise on the card: test_det_gather.py (its CUDA cases are (b)), then the record-only gather probe"
(cd cardtests && perl -e 'alarm 1800; exec @ARGV' python -m pytest -q -p no:cacheprovider -rs test_det_gather.py) > logs/cardtests.log 2>&1; prc=$?
grep -E "passed|failed|skipped|error" logs/cardtests.log | tail -2 | tee -a summary.txt
grep -q "skipped" logs/cardtests.log && grep -E "SKIPPED" logs/cardtests.log | tee -a summary.txt
[ "$prc" = 0 ] && ! grep -q "SKIPPED" logs/cardtests.log \
  || { tail -15 logs/cardtests.log; echo "PREMISE FAIL: test_det_gather.py on the card (rc $prc, or a CUDA case skipped)" | tee -a summary.txt; finish 31; }
perl -e 'alarm 900; exec @ARGV' python det/det_selfpair.py --gather-probe --out receipts/det_gather_probe.json > logs/probe.log 2>&1
grep -E "^\[det\]" logs/probe.log | tee -a summary.txt
python det/det_selfpair.py --self-test 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { echo "SELF-TEST FAIL" | tee -a summary.txt; finish 21; }

say "fetch Qwen/Qwen3-30B-A3B@ad44e777 (TC1's pin)"
fetch(){ perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-30B-A3B', revision='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39',
                        allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.jinja']))" 2>> logs/fetch.log | tail -1; }
snap=$(fetch); [ -d "$snap" ] || { echo "fetch retry" | tee -a summary.txt; snap=$(fetch); }
[ -d "$snap" ] || { tail -3 logs/fetch.log; echo "FETCH FAILED" | tee -a summary.txt; finish 22; }
echo "fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt

rc=0
for ARM in A B C Cp; do
  say "arm $ARM (its own process)"
  if [ "$ARM" = C ] || [ "$ARM" = Cp ]; then CW=":4096:8"; else CW=""; fi
  env E4B_TRAIN_DET_GATHER=1 ${CW:+CUBLAS_WORKSPACE_CONFIG=$CW} perl -e "alarm ${DET_ALARM_S:-1500}; exec @ARGV" \
    python det/det_selfpair.py --arm $ARM --model "$snap" --seq 512 --accum 4 --r 16 --alpha 16 --reps 3 \
    --out receipts/det_selfpair_$ARM.json > logs/det_$ARM.log 2>&1
  arc=$?
  grep -E "^\[det\]" logs/det_$ARM.log | tee -a summary.txt
  [ "$arc" = 0 ] || { tail -8 logs/det_$ARM.log | tee -a summary.txt; echo "arm $ARM rc=$arc" | tee -a summary.txt; rc=$arc; }
done
echo "det rc=$rc" | tee -a summary.txt
finish $rc
