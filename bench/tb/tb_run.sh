#!/bin/bash
# bench/tb/tb_run.sh -- T-B option C's one-layer prototype, BOX side (EXPLORE, never a reading; Map 1, e4b#2043 section 5).
# Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER (TC1_EXTRA_STAGE carries this file, tb_layer.py, tb_narrow_check.py,
# the ladder, and the two kernel test files), so it speaks tc1_drive's contract: the nonce handshake, summary.txt, and
# TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at the end, a refusal included. Nothing here creates,
# destroys or approves compute.
#   install gnf4 + e4b at the pins -> tripwire -> THE PREMISE ON THE CARD (rc 31 stops the box before any fetch or timing:
#   a finding, not a retry): test_fused_pad.py + test_lean_combine.py with the compiled kernels, tb_narrow_check.py
#   -> fetch Qwen3-30B-A3B at TC1's pin -> tb_layer.py's self-test -> the bench
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] tb: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
echo "EXPLORE (T-B option C one-layer prototype, Map 1): not a reading" | tee -a summary.txt
for f in tb_layer.py tb_narrow_check.py ladder.py test_fused_pad.py test_lean_combine.py; do
  [ -s $W/$f ] || { echo "STAGE MISSING: $f" | tee -a summary.txt; finish 9; }
done
mkdir -p $W/moegen $W/tb $W/cardtests && cp $W/ladder.py $W/moegen/ladder.py && cp $W/tb_layer.py $W/tb_narrow_check.py $W/tb/ \
  && cp $W/test_fused_pad.py $W/test_lean_combine.py $W/cardtests/

# floors: the model is about 61 GB on disk, and the step needs the 32 GB card
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
import lora_pad, nf4_qlora                                   # the fused pad ships in the wheel
from experts4bit_qlora.engines import combine_lean, fast     # the lean combine
assert hasattr(nf4_qlora, "FUSED_PAD_STATS") and "lean_fwd" in fast.COMBINE_STATS
print("tb tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name(),
      "sm_%d%d" % torch.cuda.get_device_capability())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt

say "premise on the card: the kernel tests with compiled Triton, then the narrowing check"
unset TRITON_INTERPRET
ct(){ (cd cardtests && perl -e 'alarm 1800; exec @ARGV' python -m pytest -q -p no:cacheprovider -rs "$@" test_fused_pad.py test_lean_combine.py); }
ct > logs/cardtests.log 2>&1; prc=$?
ct -k cuda > logs/cardtests_cuda.log 2>&1; crc=$?
ncuda=$(grep -oE "[0-9]+ passed" logs/cardtests_cuda.log | tr -dc 0-9)
echo "cardtests: all rc $prc ($(grep -oE "[0-9]+ (passed|failed|skipped|error)" logs/cardtests.log | tr '\n' ' ')); on the card rc $crc, ${ncuda:-0} passed" | tee -a summary.txt
[ "$prc" = 0 ] && [ "$crc" = 0 ] && [ "${ncuda:-0}" -gt 0 ] \
  || { tail -15 logs/cardtests.log; echo "PREMISE FAIL: kernel tests on the card -- stop before timing" | tee -a summary.txt; finish 31; }
perl -e 'alarm 1200; exec @ARGV' python tb/tb_narrow_check.py --out receipts/tb_narrow_check.json > logs/narrow.log 2>&1; nrc=$?
grep -E "^\[tb-narrow\]" logs/narrow.log | tee -a summary.txt
[ "$nrc" = 0 ] || { tail -5 logs/narrow.log; echo "PREMISE FAIL: narrowing check rc $nrc -- stop before timing" | tee -a summary.txt; finish 31; }
python tb/tb_layer.py --self-test 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { echo "SELF-TEST FAIL" | tee -a summary.txt; finish 21; }

say "fetch Qwen/Qwen3-30B-A3B@ad44e777 (TC1's pin)"
fetch(){ perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-30B-A3B', revision='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39',
                        allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.jinja']))" 2>> logs/fetch.log | tail -1; }
snap=$(fetch); [ -d "$snap" ] || { echo "fetch retry" | tee -a summary.txt; snap=$(fetch); }
[ -d "$snap" ] || { tail -3 logs/fetch.log; echo "FETCH FAILED" | tee -a summary.txt; finish 22; }
echo "fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt

say "bench: layer ${TB_LAYER:-24}, ${TB_REPS:-20} reps a block, mirrored (base t2 t1 t1t2 t1t2 t1 t2 base); the step at seq 512 x 4"
perl -e "alarm ${TB_ALARM_S:-3600}; exec @ARGV" python tb/tb_layer.py --model "$snap" --seq 512 --accum 4 --r 16 --alpha 16 \
  --layer "${TB_LAYER:-24}" --reps "${TB_REPS:-20}" --out receipts/tb_layer.json > logs/tb_layer.log 2>&1
rc=$?
grep -E "^\[tb\]" logs/tb_layer.log | tee -a summary.txt
[ "$rc" = 0 ] || tail -15 logs/tb_layer.log | tee -a summary.txt
echo "tb_layer rc=$rc" | tee -a summary.txt
finish $rc
