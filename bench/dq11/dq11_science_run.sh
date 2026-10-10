#!/bin/bash
# Real box instrument; TC1 nonce/receipt lifecycle, no provider creation or unguarded replacement draw.
if [ "${DQ11_REHEARSAL:-}" = 1 ]; then
  exec python3 "$(dirname "$0")/dq11_rehearsal.py" run
fi
case "${DQ11_REHEARSAL:-}${DQ11_REHEARSAL_TINY_MODEL:-}" in "") ;; *) exit 78;; esac
set -uo pipefail
W=${DQ11_W:-/root/tc1}; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}
case "$NONCE" in *[!a-zA-Z0-9_-]*|"") exit 78;; esac
[ ${#NONCE} -le 128 ] || exit 78
printf '%s\n' "$NONCE" > TC1_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
say(){ echo "[$(date -u +%FT%TZ)] dq11: $*" | tee -a summary.txt; }
mkdir -p logs receipts adapters data hf-cache
: > summary.txt
sha256sum -c science.sha256 > logs/science-tripwire.log 2>&1 || finish 9
mv adapter_init.safetensors adapters/ || finish 9
mv tokens.json data/ || finish 9
sha256sum -c inputs.sha256 > logs/input-tripwire.log 2>&1 || finish 9
[ -n "${E4B_SHA:-}" ] && [ -n "${TC1_DEADLINE_EPOCH:-}" ] || finish 78
case "$TC1_DEADLINE_EPOCH" in *[!0-9]*|"") finish 78;; esac
export E4B_SHA TC1_DEADLINE_EPOCH
budget_cap(){
  local requested=$1 left=$((TC1_DEADLINE_EPOCH - $(date +%s) - 300))
  [ "$left" -gt 0 ] || { say 'VOID: deadline reserve'; finish 11; }
  PHASE_CAP=$requested
  [ "$left" -ge "$requested" ] || PHASE_CAP=$left
}
budget_cap 240
perl -e 'alarm shift; exec @ARGV' "$PHASE_CAP" bash dq11_require_git.sh > logs/git-prerequisite.log 2>&1 || {
  say 'VOID: git prerequisite; no wheel/model fetch'
  finish 20
}
link=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
case "$link" in "NVIDIA GeForce RTX 5090, "3[0-3][0-9][0-9][0-9]) ;; *) say "VOID: unregistered card/VRAM $link"; finish 19;; esac
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
budget_cap 120
perl -e 'alarm shift; exec @ARGV' "$PHASE_CAP" python3 dq3_vram_probe.py > logs/vram_probe.log 2>&1; rc=$?
if [ "$rc" = 3 ]; then echo 'vram floor' > REFUSAL; finish 18; elif [ "$rc" != 0 ]; then finish 9; fi
budget_cap 60
perl -e 'alarm shift; exec @ARGV' "$PHASE_CAP" python3 dq3_egress_probe.py > logs/egress_probe.log 2>&1; rc=$?
if [ "$rc" = 4 ]; then echo 'egress' > REFUSAL; finish 14; elif [ "$rc" != 0 ]; then finish 9; fi
free_kb=$(df -Pk . | tail -1 | awk '{print $4}')
[ "$free_kb" -ge 120000000 ] || { say 'VOID: disk below 120 GB'; finish 9; }
phase(){
  local name=$1 requested=$2; shift 2
  budget_cap "$requested"
  say "$name"
  perl -e 'alarm shift; exec @ARGV' "$PHASE_CAP" "$@" > "logs/$name.log" 2>&1 || { tail -30 "logs/$name.log"; say "VOID: $name"; finish 11; }
}
VENV="$W/venv-dq11"
phase create-venv 120 python3.11 -m venv "$VENV"
export PATH="$VENV/bin:$PATH"
phase bootstrap 1800 python dq11_bootstrap.py
phase prepare 1800 python dq11_prepare.py
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader > forensics.txt
for arm in L U U0; do
  phase "proof-$arm" 900 python dq11_arm.py --kind proof --arm "$arm" --repetition 0 --out "receipts/proof-$arm.json"
done
python dq11_reduce.py receipts --initial-only > receipts/initial-gate.json 2> logs/initial-gate.log || { say 'QUALITY_FAIL/VOID: initial gate; no updates'; finish 12; }
for tag in 1-L 1-U 1-U0 2-U0 2-U 2-L; do
  rep=${tag%%-*}; arm=${tag#*-}
  phase "read-$tag" 900 python dq11_arm.py --kind read --arm "$arm" --repetition "$rep" --out "receipts/read-$tag.json"
done
python dq11_reduce.py receipts > receipts/provisional-read.json 2> logs/reduce.log || { say 'QUALITY_FAIL/VOID: no recommendation'; finish 12; }
say 'PROVISIONAL: guard teardown and instance-absence proof still required; no final position yet'
finish 0
