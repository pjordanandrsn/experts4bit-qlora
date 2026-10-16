#!/bin/bash
# Paid L-only census box worker. The guarded rental controller owns teardown.
set -uo pipefail
W=${DQ11_W:-/root/tc1}
cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}
case "$NONCE" in *[!a-zA-Z0-9_-]*|"") exit 78;; esac
[ ${#NONCE} -le 128 ] || exit 78
printf '%s\n' "$NONCE" > TC1_RUN_NONCE
mkdir -p logs receipts adapters data hf-cache
: > summary.txt
say(){ echo "[$(date -u +%FT%TZ)] dq11-l-census: $*" | tee -a summary.txt; }
finish(){
  local rc=$1
  printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"
  [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"
  : > "TP_DONE.$NONCE"
  exit "$rc"
}
[ "${E4B_SHA:-}" = 13b55990433f357d63232f75926ce19f5bda5b42 ] || {
  say 'CENSUS_REFUSAL: science source is not 13b'; finish 78;
}
case "${TC1_DEADLINE_EPOCH:-}" in *[!0-9]*|"") say 'CENSUS_REFUSAL: no guard deadline'; finish 78;; esac
sha256sum -c science.sha256 > logs/science-tripwire.log 2>&1 || finish 9
sha256sum -c dq11-l-census-stage.sha256 > logs/instrument-tripwire.log 2>&1 || finish 9
cp dq11-l-census-stage.sha256 receipts/expected-instrument-and-operational.sha256 || finish 9
sha256sum dq11_l_census_run.py dq11_l_census_capture.py dq11_l_census_reduce.py dq11_l_census_launch.py dq11_l_census_box.sh dq11_l_census_drive.sh dq11_wheel_retry.py dq11_census_transport.py > receipts/instrument-and-operational.sha256 || finish 9
mv adapter_init.safetensors adapters/ || finish 9
mv tokens.json data/ || finish 9
sha256sum -c inputs.sha256 > logs/input-tripwire.log 2>&1 || finish 9
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF DQ11_REHEARSAL DQ11_REHEARSAL_TINY_MODEL
export E4B_SHA TC1_DEADLINE_EPOCH
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
budget_cap(){
  local wanted=$1 left=$((TC1_DEADLINE_EPOCH - $(date +%s) - 300))
  [ "$left" -gt 0 ] || { say 'CENSUS_REFUSAL: five-minute teardown reserve exhausted'; finish 11; }
  PHASE_CAP=$wanted
  [ "$left" -ge "$wanted" ] || PHASE_CAP=$left
}
phase(){
  local name=$1 wanted=$2; shift 2
  budget_cap "$wanted"
  say "$name"
  python3 dq11_census_transport.py phase --seconds "$PHASE_CAP" --deadline "$TC1_DEADLINE_EPOCH" --name "$name" "$@" > "logs/$name.log" 2>&1 || {
    tail -30 "logs/$name.log"
    say "CENSUS_REFUSAL: $name"
    finish 11
  }
}
budget_cap 240
python3 dq11_census_transport.py phase --seconds "$PHASE_CAP" --deadline "$TC1_DEADLINE_EPOCH" --name git-prerequisite bash dq11_require_git.sh > logs/git-prerequisite.log 2>&1 || {
  say 'CENSUS_REFUSAL: git prerequisite'; finish 20;
}
[[ "${DQ11_REQUIRED_DRIVER:-}" =~ ^[0-9]{3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]] || {
  say 'CENSUS_REFUSAL: exact manifest driver missing'; finish 19;
}
card=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
python3 - "$card" "$DQ11_REQUIRED_DRIVER" <<'PY' > logs/hardware-precheck.log 2>&1 || { say 'CENSUS_REFUSAL: GPU/driver'; finish 19; }
import re, sys
parts = [x.strip() for x in sys.argv[1].split(',')]
assert len(parts) == 3 and parts[0] == 'NVIDIA GeForce RTX 5090'
assert 30000 <= int(parts[1]) <= 34000
assert re.fullmatch(r'[0-9]{3}\.[0-9]{1,3}(?:\.[0-9]{1,3})?', parts[2])
assert int(parts[2].split('.')[0]) >= 580
assert parts[2] == sys.argv[2]
print('one normal RTX 5090, driver', parts[2])
PY
phase vram-probe 120 python3 dq3_vram_probe.py
phase egress-probe 60 python3 dq3_egress_probe.py
free_kb=$(df -Pk . | tail -1 | awk '{print $4}')
[ "$free_kb" -ge 120000000 ] || { say 'CENSUS_REFUSAL: disk below 120 GB'; finish 9; }
VENV="$W/venv-dq11"
phase create-venv 120 python3.11 -m venv "$VENV"
export PATH="$VENV/bin:$PATH"
phase bootstrap 900 python dq11_wheel_retry.py
phase cuda-cupti 90 python -c 'import torch; assert torch.__version__ == "2.12.1+cu130" and torch.version.cuda == "13.0"; assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (12, 0); print(torch.cuda.get_device_name(), torch.__version__, torch.cuda.get_device_capability())'
phase prepare 600 python dq11_prepare.py
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader > forensics.txt
phase proof 240 python dq11_l_census_run.py --directory "$W" --science "$W" --kind proof --trace receipts/proof-trace.json --out receipts/proof.json
phase reference 300 python dq11_l_census_run.py --directory "$W" --science "$W" --kind reference --proof receipts/proof.json --trace receipts/reference-unused.json --out receipts/reference.json
phase capture 300 python dq11_l_census_run.py --directory "$W" --science "$W" --kind capture --proof receipts/proof.json --trace receipts/capture-trace.json --out receipts/capture.json
phase reduce 120 python dq11_l_census_reduce.py --trace receipts/capture-trace.json --proof-trace receipts/proof-trace.json --reference receipts/reference.json --capture receipts/capture.json --proof receipts/proof.json --normalized-out receipts/normalized.json --out receipts/decision.json
phase packet-size 120 python dq11_l_census_launch.py size --packet "$W" --out /root/dq11-l-census-box-size-"$NONCE".json
say 'PROVISIONAL: only the guarded controller can close transfer, invoice and teardown'
finish 0
