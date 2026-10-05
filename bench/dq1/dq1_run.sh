#!/bin/bash
# bench/dq1/dq1_run.sh -- lane DQ1, BOX side (bench/dq1/DQ1-PREREG.md). Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER
# (TC1_EXTRA_STAGE carries this file, dq1_census.py and dq1_reduce.py), so it speaks tc1_drive's contract: the nonce handshake
# (TC1_RUN_NONCE within 30 s), summary.txt one line per step, TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at
# the end -- a refusal writes them too. No checkpoint: the census reads Qwen3-32B's linear SHAPES on random NF4 weights.
# Nothing here creates, destroys or approves compute.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
say(){ echo "[$(date -u +%FT%TZ)] dq1: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}
# grouped-nf4-gemm v0.39.0 (`git rev-parse v0.39.0^{commit}`): the release whose GNF4_TRAIN_GEMM=auto takes the dense route at G=1.
DQ1_GNF4_SHA=a5edec8789735bff1c0da4708ae5fc93260a1410
DQ1_BNB_VER=0.50.2

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: grouped-nf4-gemm @$DQ1_GNF4_SHA, bitsandbytes $DQ1_BNB_VER (torch from the image)"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$DQ1_GNF4_SHA" "bitsandbytes==$DQ1_BNB_VER" > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, torch, bitsandbytes
import nf4_grouped, nf4_qlora, nf4_route                       # the modules the census measures
got = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
assert got == "a5edec8789735bff1c0da4708ae5fc93260a1410", f"grouped-nf4-gemm installed {got}"
assert md.version("bitsandbytes") == "0.50.2", md.version("bitsandbytes")
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability()
assert nf4_route.train_gemm_route(torch.device("cuda", 0), 1) == "dense", "premise: auto must take the dense route at G=1 here"
print("dq1 tripwire OK", md.version("grouped-nf4-gemm"), bitsandbytes.__version__, torch.__version__, torch.version.cuda,
      torch.cuda.get_device_name(), cap)
PYT
cat logs/tripwire.log | tee -a summary.txt
python dq1_reduce.py --self-test > logs/reduce_selftest.log 2>&1 \
  || { cat logs/reduce_selftest.log; echo "REDUCER SELF-TEST FAIL" | tee -a summary.txt; finish 9; }
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt

say "census"
perl -e 'alarm 2100; exec @ARGV' python dq1_census.py --out receipts/dq1_census.json > logs/census.log 2>&1
rc=$?; echo "census rc=$rc $(tail -1 logs/census.log | cut -c1-200)" | tee -a summary.txt
[ "$rc" = 0 ] || { tail -30 logs/census.log; finish 11; }
python dq1_reduce.py receipts/dq1_census.json > receipts/dq1_read.json 2> logs/reduce.log \
  || { cat logs/reduce.log; echo "REDUCE FAIL" | tee -a summary.txt; finish 12; }
python -c "import json; v=json.load(open('receipts/dq1_read.json'))['verdicts']; print('DQ1 read:', v)" | tee -a summary.txt
finish 0
