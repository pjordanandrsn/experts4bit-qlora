#!/bin/bash
# bench/dq3/dq3_run.sh -- lane DQ3, BOX side (bench/dq3/DQ3-PREREG.md, Amendments 0-1). Started by bench/tc1/tc1_drive.sh as its
# TC1_RUNNER (TC1_EXTRA_STAGE carries this file, dq3_vram_probe.py, dq3_arm.py and dq3_reduce.py), so it speaks tc1_drive's contract: the nonce
# handshake (TC1_RUN_NONCE within 30 s), summary.txt one line per step, TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> /
# TP_DONE.<nonce> at the end -- a refusal writes them too. No checkpoint download: the subject is Qwen3-32B's architecture
# with random NF4 weights (Amendment 1). Six arms, one process each, palindrome R S S0 S0 S R. Nothing here creates,
# destroys or approves compute.
set -uo pipefail
W=${DQ3_W:-/root/tc1}; cd "$W" || exit 9   # DQ3_W: tests only
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
say(){ echo "[$(date -u +%FT%TZ)] dq3: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}
DQ3_GNF4_SHA=a5edec8789735bff1c0da4708ae5fc93260a1410

link=$(nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits 2>&1 | head -1)
echo "link: $link" | tee -a summary.txt
case "$link" in *"RTX 5090, 5, 16") ;; *) echo "HOST REFUSED: not an RTX 5090 on PCIe gen 5 x16 ($link)" | tee -a summary.txt; finish 13;; esac
# Host floor (rc 18, rent.py's machine evidence): the GPU must hand out the subject's memory. dq3-5090-1's host refused
# the first 2.90 GiB allocation with 30.85 GiB free; see dq3_vram_probe.py. Before any install, with the image's torch.
# Only the probe's OOM exit (3) names the host; any other failure (no device, no kernels for the card, an exception) is
# a harness error (9) and excludes nothing.
python dq3_vram_probe.py > logs/vram_probe.log 2>&1; prc=$?
if [ "$prc" = 3 ]; then tail -2 logs/vram_probe.log; echo "refused: vram floor" > REFUSAL
  echo "BOX_REFUSED vram: $(tail -1 logs/vram_probe.log | cut -c1-200)" | tee -a summary.txt; finish 18
elif [ "$prc" != 0 ]; then tail -5 logs/vram_probe.log; echo "VRAM PROBE ERROR rc=$prc (not a host refusal)" | tee -a summary.txt; finish 9; fi
tail -1 logs/vram_probe.log | tee -a summary.txt

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: experts4bit-qlora @$E4B_SHA, grouped-nf4-gemm @$DQ3_GNF4_SHA, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$DQ3_GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "bitsandbytes==0.50.2" "transformers==5.18.0" "peft==0.21.2" accelerate safetensors > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, inspect, json, os, torch
from experts4bit_qlora.engines.dense_offload import enable_dense_offload, train_schedule   # the subject's opt-in
got = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
assert got == os.environ["E4B_SHA"], f"experts4bit-qlora installed {got} != {os.environ['E4B_SHA']}"
assert "train_prefetch" in inspect.signature(enable_dense_offload).parameters
for d, v in (("bitsandbytes", "0.50.2"), ("transformers", "5.18.0"), ("peft", "0.21.2")):
    assert md.version(d) == v, f"{d} {md.version(d)} != {v}"
assert torch.cuda.is_available()
print("dq3 tripwire OK", got[:8], torch.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
python dq3_reduce.py --self-test > logs/reduce_selftest.log 2>&1 \
  || { cat logs/reduce_selftest.log; echo "REDUCER SELF-TEST FAIL" | tee -a summary.txt; finish 9; }
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF= TOKENIZERS_PARALLELISM=false
k=0; files=""
for arm in R S S0 S0 S R; do
  k=$((k + 1)); out="receipts/arm${k}_${arm}.json"
  say "arm $k: $arm"
  perl -e 'alarm 1500; exec @ARGV' python dq3_arm.py --arm "$arm" --out "$out" > "logs/arm${k}_${arm}.log" 2>&1
  rc=$?; echo "arm $k $arm rc=$rc $(tail -1 "logs/arm${k}_${arm}.log" | cut -c1-200)" | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -30 "logs/arm${k}_${arm}.log"; finish 11; }
  files="$files $out"
done
python dq3_reduce.py $files > receipts/dq3_read.json 2> logs/reduce.log \
  || { cat logs/reduce.log; echo "REDUCE FAIL" | tee -a summary.txt; finish 12; }
python -c "import json; d=json.load(open('receipts/dq3_read.json')); r=d['readings']; print('DQ3 read:', d['verdicts'], {k: r.get(k) for k in ('S_over_R','S0_over_R','saving_fraction','R_self_pair')})" | tee -a summary.txt
finish 0
