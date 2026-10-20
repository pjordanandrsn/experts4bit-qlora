#!/bin/bash
# DQ11 MLP-adapter BUILD Phase 2 box worker (DQ11-MLP-BUILD-AMENDMENT-5.md): one RTX 5090 at the science lock.
# Started by bench/tc1/tc1_drive.sh as TC1_RUNNER. The guarded rental controller owns teardown.
# The census box's bootstrap and prepare by file; then one process per part (dq11_mlp_phase2_run.py --target 5090).
# The timed block's records are on the box (and in the controller's partial fetch) before D0 starts; a drift arm that fails,
# or that the guard cannot fit, reads "drift: not read" and never changes the box's exit.
set -uo pipefail
W=${DQ11_W:-/root/tc1}
cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}
case "$NONCE" in *[!a-zA-Z0-9_-]*|"") exit 78;; esac
[ ${#NONCE} -le 128 ] || exit 78
printf '%s\n' "$NONCE" > TC1_RUN_NONCE
mkdir -p logs receipts/phase2 adapters data hf-cache
: > summary.txt
say(){ echo "[$(date -u +%FT%TZ)] dq11-mlp-phase2: $*" | tee -a summary.txt; }
finish(){
  local rc=$1
  printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"
  [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"
  : > "TP_DONE.$NONCE"
  exit "$rc"
}
case "${E4B_SHA:-}" in *[!0-9a-f]*|"") say 'PHASE2_REFUSAL: no e4b head'; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say 'PHASE2_REFUSAL: e4b head is not 40-hex'; finish 78; }
case "${TC1_DEADLINE_EPOCH:-}" in *[!0-9]*|"") say 'PHASE2_REFUSAL: no guard deadline'; finish 78;; esac
sha256sum -c science.sha256 > logs/science-tripwire.log 2>&1 || finish 9
sha256sum -c dq11-mlp-phase2-stage.sha256 > logs/instrument-tripwire.log 2>&1 || finish 9
cp dq11-mlp-phase2-stage.sha256 receipts/expected-instrument.sha256 || finish 9
mv adapter_init.safetensors adapters/ || finish 9
mv tokens.json data/ || finish 9
sha256sum -c inputs.sha256 > logs/input-tripwire.log 2>&1 || finish 9
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF DQ11_REHEARSAL DQ11_REHEARSAL_TINY_MODEL CUBLAS_WORKSPACE_CONFIG E4B_DENSE_MLP_LORA_FUSE
# The rental is this read's claim on its card: the runner's claim check reads this.
export E4B_SHA TC1_DEADLINE_EPOCH DQ11_GPU_CLAIM_CONFIRMED=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
left(){ echo $((TC1_DEADLINE_EPOCH - $(date +%s) - 300)); }
budget_cap(){
  local wanted=$1 l; l=$(left)
  [ "$l" -gt 0 ] || { say 'PHASE2_REFUSAL: five-minute teardown reserve exhausted'; finish 11; }
  PHASE_CAP=$wanted
  [ "$l" -ge "$wanted" ] || PHASE_CAP=$l
}
run_phase(){
  local name=$1 wanted=$2; shift 2
  budget_cap "$wanted"
  say "$name"
  python3 dq11_census_transport.py phase --seconds "$PHASE_CAP" --deadline "$TC1_DEADLINE_EPOCH" --name "$name" "$@" > "logs/$name.log" 2>&1
}
phase(){
  run_phase "$@" || { tail -30 "logs/$1.log"; say "PHASE2_REFUSAL: $1"; finish 11; }
}
# The guard fit, before anything installs (Amendment 5 section 8): the registered part seconds x 1.25 must fit what is left.
need=$(python3 dq11_mlp_phase2_run.py --target 5090 --part fit --directory "$W" --cache "$W" --out receipts/phase2) || {
  say 'PHASE2_REFUSAL: no fit'; finish 41;
}
[ "$need" -le "$(left)" ] || { say "PHASE2_REFUSAL: the fit needs ${need} s and $(left) s are left"; finish 41; }
say "fit: ${need} s needed, $(left) s left"
budget_cap 240
python3 dq11_census_transport.py phase --seconds "$PHASE_CAP" --deadline "$TC1_DEADLINE_EPOCH" --name git-prerequisite bash dq11_require_git.sh > logs/git-prerequisite.log 2>&1 || {
  say 'PHASE2_REFUSAL: git prerequisite'; finish 20;
}
[[ "${DQ11_REQUIRED_DRIVER:-}" =~ ^[0-9]{3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]] || {
  say 'PHASE2_REFUSAL: exact manifest driver missing'; finish 19;
}
card=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
python3 - "$card" "$DQ11_REQUIRED_DRIVER" <<'PY' > logs/hardware-precheck.log 2>&1 || { say 'PHASE2_REFUSAL: GPU/driver'; finish 19; }
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
[ "$free_kb" -ge 120000000 ] || { say 'PHASE2_REFUSAL: disk below 120 GB'; finish 9; }
VENV="$W/venv-dq11"
phase create-venv 120 python3.11 -m venv "$VENV"
export PATH="$VENV/bin:$PATH"
phase bootstrap 900 python dq11_wheel_retry.py
phase cuda-cupti 90 python -c 'import torch; assert torch.__version__ == "2.12.1+cu130" and torch.version.cuda == "13.0"; assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (12, 0); print(torch.cuda.get_device_name(), torch.__version__, torch.cuda.get_device_capability())'
phase prepare 600 python dq11_prepare.py
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader > forensics.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P2="python dq11_mlp_phase2_run.py --target 5090 --directory $W --cache /root/.cache/dq11-wheels --out $W/receipts/phase2"
phase stage-check 300 $P2 --part stage-check
phase b-read 3600 $P2 --part b-read
for arm in F0a F1a F1b F0b F0c F1c F1d F0d; do phase "$arm" 900 $P2 --part "$arm"; done
say 'timed block written; waiting two polls for the controller partial fetch before any drift arm'
: > receipts/phase2/TIMED_DONE
sleep 150
for arm in D0 D1; do
  if [ "$(left)" -lt 900 ]; then say "drift: not read ($arm: the guard cannot fit it)"; break; fi
  run_phase "$arm" 600 $P2 --part "$arm" || say "drift: not read ($arm failed, see logs/$arm.log)"
done
phase reduce 300 $P2 --part reduce
say "$(tail -1 logs/reduce.log)"
say 'PROVISIONAL: only the guarded controller can close transfer, invoice and teardown'
finish 0
