#!/bin/bash
# bench/dq2/dq2_run.sh -- lane DQ2, BOX side (bench/dq2/DQ2-PREREG.md). Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER
# (TC1_EXTRA_STAGE carries this file, dq2_layer.py, dq2_reduce.py and DQ1's dq1_census.py, whose warm-up and forensics the
# layer census imports), so it speaks tc1_drive's contract: the nonce handshake (TC1_RUN_NONCE within 30 s), summary.txt one
# line per step, TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at the end -- a refusal writes them too.
# No checkpoint: one Qwen3-32B decoder layer is built from its config with random weights. Nothing here creates, destroys or
# approves compute.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
say(){ echo "[$(date -u +%FT%TZ)] dq2: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}

# ---- the link first: the lane's subject is a PCIe gen 5 x16 host; anything else is refused before any install (rc 13)
link=$(nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits 2>&1 | head -1)
echo "link: $link" | tee -a summary.txt
case "$link" in *"RTX 5090, 5, 16") ;; *) echo "HOST REFUSED: not an RTX 5090 on PCIe gen 5 x16 ($link)" | tee -a summary.txt; finish 13;; esac

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: grouped-nf4-gemm v0.39.0 (DQ1's census imports it), bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@a5edec8789735bff1c0da4708ae5fc93260a1410" \
  "bitsandbytes==0.50.2" "transformers==5.18.0" "peft==0.21.2" accelerate safetensors > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, torch, bitsandbytes, transformers, peft
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding   # the subject
from peft import inject_adapter_in_model                                                      # the adapter path
for d, v in (("bitsandbytes", "0.50.2"), ("transformers", "5.18.0"), ("peft", "0.21.2")):
    assert md.version(d) == v, f"{d} {md.version(d)} != {v}"
assert torch.cuda.is_available()
print("dq2 tripwire OK", torch.__version__, torch.version.cuda, bitsandbytes.__version__, transformers.__version__,
      peft.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
python dq2_reduce.py --self-test > logs/reduce_selftest.log 2>&1 \
  || { cat logs/reduce_selftest.log; echo "REDUCER SELF-TEST FAIL" | tee -a summary.txt; finish 9; }
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt

say "layer census"
perl -e 'alarm 2100; exec @ARGV' python dq2_layer.py --out receipts/dq2_layer.json > logs/layer.log 2>&1
rc=$?; echo "layer census rc=$rc $(tail -1 logs/layer.log | cut -c1-200)" | tee -a summary.txt
[ "$rc" = 0 ] || { tail -30 logs/layer.log; finish 11; }
python dq2_reduce.py receipts/dq2_layer.json > receipts/dq2_read.json 2> logs/reduce.log \
  || { cat logs/reduce.log; echo "REDUCE FAIL" | tee -a summary.txt; finish 12; }
python -c "import json; d=json.load(open('receipts/dq2_read.json')); print('DQ2 read:', d['verdicts'], 'break-even', d['break_even'])" | tee -a summary.txt
finish 0
