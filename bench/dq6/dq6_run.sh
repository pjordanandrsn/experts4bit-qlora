#!/bin/bash
# bench/dq6/dq6_run.sh -- lane DQ6, BOX side (bench/dq6/DQ6-PREREG.md): DQ4's capacity read on a 24 GB RTX 4090. This is
# bench/dq4/dq4_run.sh with the registered differences only (tests/test_dq6_lane.py pins them): the card gate (a 24 GB
# RTX 4090, rc 19 otherwise), DQ3's VRAM probe at a 24 GB card's floor (dq6_vram_probe.py), the ladder (start 512, step
# 512) and the reducer (dq6_reduce.py: DQ4's rule, registered device RTX 4090). Started by bench/tc1/tc1_drive.sh as its
# TC1_RUNNER (TC1_EXTRA_STAGE carries this file, dq6_vram_probe.py, dq6_reduce.py, DQ4's dq4_cap.py and dq4_reduce.py,
# bench/dq3/dq3_arm.py and DQ3's two host probes), so it speaks tc1_drive's contract. Configurations, as DQ4:
#   c_def  chunked loss, default allocator         -- GRADED: ladders R, S; fresh confirmations at L* and the first OOM
#   c_exp  chunked loss, expandable_segments:True  -- secondary verdict, same shape
#   s_def  stock loss, default allocator           -- descriptive: ladders only
# Nothing here creates, destroys or approves compute.
set -uo pipefail
W=${DQ6_W:-/root/tc1}; cd "$W" || exit 9   # DQ6_W: tests only
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
say(){ echo "[$(date -u +%FT%TZ)] dq6: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}
DQ4_GNF4_SHA=a5edec8789735bff1c0da4708ae5fc93260a1410
START=512; STEP=512   # a 24 GB card: R is predicted near 1.5-2k tokens, so the ladder starts low and steps finely
# seconds a configuration may need (2 builds + ladders + 4 confirmations, from DQ3's 5090 build/step times); skip if short
NEED_C=1800; NEED_S=600
left(){ if [ -n "${TC1_DEADLINE_EPOCH:-}" ]; then echo $(( TC1_DEADLINE_EPOCH - $(date +%s) - 300 )); else echo 999999; fi; }

link=$(nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits 2>&1 | head -1)
echo "link: $link" | tee -a summary.txt
# The CARD, not the link: capacity does not depend on the link (DQ4/DQ5), which is recorded, not gated. Exactly a 24 GB
# 4090 (memory.total 23000-24999 MiB): a 48 GB-modded 4090 also reports "RTX 4090". Out of band -> rc 19, a code adertha
# does not admit as machine evidence (#1216): a good host for another lane must not be excluded by this one.
case "$link" in "NVIDIA GeForce RTX 4090, 2"[34][0-9][0-9][0-9]", "*) ;; *) echo "OUT OF BAND: not a 24 GB RTX 4090 ($link)" | tee -a summary.txt; finish 19;; esac
# Host floor (rc 18) and egress (rc 14): DQ3's probes, unchanged (#1171, #1173). Only exit 3 / 4 name the host.
python dq6_vram_probe.py > logs/vram_probe.log 2>&1; prc=$?
if [ "$prc" = 3 ]; then echo "refused: vram floor" > REFUSAL; echo "BOX_REFUSED vram: $(tail -1 logs/vram_probe.log | cut -c1-200)" | tee -a summary.txt; finish 18
elif [ "$prc" != 0 ]; then tail -5 logs/vram_probe.log; echo "VRAM PROBE ERROR rc=$prc (not a host refusal)" | tee -a summary.txt; finish 9; fi
tail -1 logs/vram_probe.log | tee -a summary.txt
python dq3_egress_probe.py > logs/egress_probe.log 2>&1; erc=$?
if [ "$erc" = 4 ]; then echo "refused: egress" > REFUSAL; echo "BOX_REFUSED egress: $(tail -1 logs/egress_probe.log | cut -c1-200)" | tee -a summary.txt; finish 14
elif [ "$erc" != 0 ]; then tail -3 logs/egress_probe.log; echo "EGRESS PROBE ERROR rc=$erc (not a host refusal)" | tee -a summary.txt; finish 9; fi
tail -1 logs/egress_probe.log | tee -a summary.txt

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: experts4bit-qlora @$E4B_SHA, grouped-nf4-gemm @$DQ4_GNF4_SHA, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$DQ4_GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "bitsandbytes==0.50.2" "transformers==5.18.0" "peft==0.21.2" accelerate safetensors > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, inspect, json, os, torch
from experts4bit_qlora.engines.dense_offload import _bnb_mirror_mismatches, enable_dense_offload
from experts4bit_qlora.engines.chunked_lm_loss import SUPPORTED
got = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
assert got == os.environ["E4B_SHA"], f"experts4bit-qlora installed {got} != {os.environ['E4B_SHA']}"
assert "train_prefetch" in inspect.signature(enable_dense_offload).parameters
assert _bnb_mirror_mismatches() == [], "the late-bound backward would not engage: bnb source differs from the pins"
assert "Qwen3ForCausalLM" in SUPPORTED, "the chunked loss does not cover dense Qwen3 (#1193)"
for d, v in (("bitsandbytes", "0.50.2"), ("transformers", "5.18.0"), ("peft", "0.21.2")):
    assert md.version(d) == v, f"{d} {md.version(d)} != {v}"
assert torch.cuda.is_available()
print("dq6 tripwire OK", got[:8], torch.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
python dq6_reduce.py --self-test > logs/reduce_selftest.log 2>&1 \
  || { cat logs/reduce_selftest.log; echo "REDUCER SELF-TEST FAIL" | tee -a summary.txt; finish 9; }
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep "Model name" | tee -a forensics.txt; grep -E "MemTotal|MemAvailable" /proc/meminfo | tee -a forensics.txt
export TOKENIZERS_PARALLELISM=false

cap(){   # cap <tag> <loss> <alloc:default|expandable> <arm> <mode> [seq]
  local tag=$1 loss=$2 alloc=$3 arm=$4 mode=$5 seq=${6:-0} out="receipts/$1.json" rc
  local conf=""; [ "$alloc" = expandable ] && conf="expandable_segments:True"
  say "$tag"
  PYTORCH_CUDA_ALLOC_CONF=$conf perl -e 'alarm 2400; exec @ARGV' python dq4_cap.py --arm "$arm" --mode "$mode" --loss "$loss" \
    --start "$START" --step "$STEP" --seq "$seq" --out "$out" > "logs/$tag.log" 2>&1
  rc=$?; echo "$tag rc=$rc $(tail -1 "logs/$tag.log" | cut -c1-200)" | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -30 "logs/$tag.log"; finish 11; }
}
field(){ python -c "import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print('' if v is None else v)" "$1" "$2"; }

for spec in "c_def chunked default 1" "c_exp chunked expandable 1" "s_def stock default 0"; do
  set -- $spec; cfg=$1; loss=$2; alloc=$3; confirm=$4
  need=$NEED_C; [ "$confirm" = 1 ] || need=$NEED_S
  if [ "$(left)" -lt "$need" ]; then echo "SKIPPED $cfg: $(left) s left < $need s needed" | tee -a summary.txt; continue; fi
  for arm in R S; do cap "${cfg}_${arm}_ladder" "$loss" "$alloc" "$arm" ladder; done
  if [ "$confirm" = 1 ]; then
    for arm in R S; do
      ok=$(field "receipts/${cfg}_${arm}_ladder.json" max_ok); oom=$(field "receipts/${cfg}_${arm}_ladder.json" first_oom)
      [ -n "$ok" ] && cap "${cfg}_${arm}_ok" "$loss" "$alloc" "$arm" confirm "$ok"
      [ -n "$oom" ] && cap "${cfg}_${arm}_oom" "$loss" "$alloc" "$arm" confirm "$oom"
    done
  fi
done
python dq6_reduce.py receipts > receipts/dq6_read.json 2> logs/reduce.log \
  || { cat logs/reduce.log; echo "REDUCE FAIL" | tee -a summary.txt; finish 12; }
python -c "import json; d=json.load(open('receipts/dq6_read.json')); print('DQ6 read:', d['verdicts'], {c: {k: v.get(k) for k in ('L_R','L_S','G')} for c, v in d['configs'].items()})" | tee -a summary.txt
finish 0
