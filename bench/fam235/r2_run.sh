#!/bin/bash
# R2-Mixtral's box runner (bench/fam235/PREREG-r2-mixtral.md): a copy of Q235's runner (bench/q235/q235_run.sh), which it follows
# line for line but for the declared substitutions in bench/fam235/r2_drift.py, checked by tests/test_r2_harness.py against the
# Q235 commit it cites. It runs Q235's own box, tokens, fetch, bootstrap and host-rate tools unchanged; R2's are the lock
# (model_files.json, Mixtral-8x22B at its revision), the closure (r2.sha256) and the CPU rehearsal's tiny Mixtral (r2_tiny.py).
#
# Until the amendment that registers the box pins its class, floors and phase caps, the science mode refuses: Q235_REHEARSAL=a2000
# (a depth-truncated slice on the lane's RTX A2000) and Q235_REHEARSAL=cpu (a tiny Mixtral) are the modes this script runs. Neither
# is ever a scientific draw: receipts/run.json says science_eligible false.
set -uo pipefail
W=${Q235_W:-/root/tc1}; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}
case "$NONCE" in *[!a-zA-Z0-9_-]*|"") exit 78;; esac
[ ${#NONCE} -le 128 ] || exit 78
printf '%s\n' "$NONCE" > TC1_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
say(){ echo "[$(date -u +%FT%TZ)] r2: $*" | tee -a summary.txt; }
mkdir -p logs receipts
: > summary.txt
MODE=${Q235_REHEARSAL:-science}
case "$MODE" in science|a2000|cpu) ;; *) say "refusing: Q235_REHEARSAL=$MODE"; finish 78;; esac
sha256sum -c r2.sha256 > logs/closure-tripwire.log 2>&1 || { say 'VOID: the staged closure does not match r2.sha256'; finish 9; }
for v in E4B_SHA GNF4_SHA TC1_DEADLINE_EPOCH; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$TC1_DEADLINE_EPOCH" in *[!0-9]*) finish 78;; esac
export E4B_SHA GNF4_SHA TC1_DEADLINE_EPOCH
ELIGIBLE=false; [ "$MODE" = science ] && ELIGIBLE=true
printf '{"schema": "r2-run/1", "mode": "%s", "science_eligible": %s, "e4b_sha": "%s", "gnf4_sha": "%s", "nonce": "%s", "start_utc": "%s"}\n' \
  "$MODE" "$ELIGIBLE" "$E4B_SHA" "$GNF4_SHA" "$NONCE" "$(date -u +%FT%TZ)" > receipts/run.json

budget_cap(){
  local requested=$1 left=$((TC1_DEADLINE_EPOCH - $(date +%s) - 300))
  [ "$left" -gt 0 ] || { say 'VOID: deadline reserve reached'; finish 11; }
  PHASE_CAP=$requested
  [ "$left" -ge "$requested" ] || PHASE_CAP=$left
}
phase(){   # name requested-seconds command... ; sets PHASE_RC; records the phase's timing whatever it returns
  local name=$1 requested=$2 t0 t1; shift 2
  budget_cap "$requested"
  say "$name (cap ${PHASE_CAP} s of ${requested} s)"
  t0=$(date +%s)
  perl -e 'alarm shift; exec @ARGV' "$PHASE_CAP" "$@" > "logs/$name.log" 2>&1; PHASE_RC=$?
  t1=$(date +%s)
  printf '{"phase": "%s", "requested_s": %d, "cap_s": %d, "elapsed_s": %d, "left_s": %d, "rc": %d}\n' \
    "$name" "$requested" "$PHASE_CAP" $((t1 - t0)) $((TC1_DEADLINE_EPOCH - t1)) "$PHASE_RC" >> receipts/phase-timings.jsonl
}
must(){ phase "$@"; [ "$PHASE_RC" = 0 ] || { tail -30 "logs/$1.log"; say "VOID: $1 (rc $PHASE_RC)"; finish 11; }; }

# The registered constants live here, in the sealed closure: tc1_drive.sh forwards only its own TC1_* list, so nothing below can
# be changed from the controller's environment. A rehearsal may override the caps and the interpreter (Q235_CAP_*, Q235_PYTHON).
# The phase caps (seconds) are gate 2's: each fitted from the rehearsal's measured phases, scaled to the registered model and class.
# Fitted (r2_fit.py, R2 Amendment 3) from R2's A2000 rehearsal at a3e73429, slowed to the floors and the RTX 5090 spread: for a
# host listing at least 500 Mbps, box 2,389 s and repeat 1,330 s scaled (x 56/4), fetch 4,715 s; each cap is that x 1.5, to the minute.
# R2 Amendment 4 re-cites Q235 and keeps these caps: Q235 Amendment 7's box refit (13,847 s) times Q235's shape-B box with its
# timed MoE pass, which R2's box does not run (shape A, no --time-moe: R2's extra= line below).
CAP_GIT=240 CAP_VENV=120 CAP_INSTALL=360 CAP_HOSTRATE=600 CAP_FETCH=7080 CAP_TOKENS=300 CAP_BOX=3600 CAP_REPEAT=2040
# The host-rate floors (gate 2's premise): Q235's sealed cold ones, measured on the same RTX A2000 host (R2 Amendment 3: R2's own leg
# read its fresh slice warm), its pinned H2D (PCIe gen 3 x8), host-copy rate and load's effective read rate. The box measures all
# three (the disk with O_DIRECT) before the checkpoint fetch and refuses below any. A rehearsal records only.
MIN_H2D_GBS=5.8 MIN_DISK_GBS=0.743 MIN_HOST_COPY_GBS=5.14
# The pull-rate floor (GB/s): the rate at which the checkpoint, hashed at the rehearsal's rate, still fits CAP_FETCH / 1.5. The fetch
# times its first slice (the small files and 8 shards, ~32 GB) with its own client and refuses (14) below it, before pulling the rest.
MIN_FETCH_GBS=0.063
ROWS=8          # TC1's field-text rows in the one micro-batch (the 30B calibration's and gate 1's); the registering amendment fixes it
PY=python3.11
if [ "$MODE" != science ]; then
  CAP_FETCH=${Q235_CAP_FETCH:-$CAP_FETCH} CAP_BOX=${Q235_CAP_BOX:-$CAP_BOX} CAP_REPEAT=${Q235_CAP_REPEAT:-$CAP_REPEAT}
  PY=${Q235_PYTHON:-$PY}
fi

must require-git "$CAP_GIT" bash dq11_require_git.sh

# Host floors, before any fetch (PREREG "Refusal points" 4-5; the card class is the registering amendment's).
case "$MODE" in
  science) CARD_RE='^NVIDIA GeForce RTX 5090'; MIN_VRAM_MIB=32000; MIN_RAM_GB=134; MIN_DISK_GB=350 ;;   # R2 Amendments 2 and 3
  a2000)   CARD_RE='^NVIDIA RTX A2000 12GB$'; MIN_VRAM_MIB=11000; MIN_RAM_GB=64; MIN_DISK_GB=60 ;;
  cpu)     CARD_RE=''; MIN_VRAM_MIB=0; MIN_RAM_GB=8; MIN_DISK_GB=10 ;;
esac
if [ -n "$CARD_RE" ]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit \
    --format=csv,noheader > receipts/card.txt 2>&1 || { say 'VOID: nvidia-smi failed'; finish 9; }
  [ "$(wc -l < receipts/card.txt)" -eq 1 ] || { say 'refusing: not exactly one GPU'; echo 'gpu count' > REFUSAL; finish 18; }
  name=$(cut -d, -f1 receipts/card.txt); vram=$(cut -d, -f2 receipts/card.txt | tr -dc 0-9)
  echo "$name" | grep -Eq "$CARD_RE" || { say "refusing: card $name is not the registered class"; echo 'card class' > REFUSAL; finish 18; }
  [ "${vram:-0}" -ge "$MIN_VRAM_MIB" ] || { say "refusing: ${vram} MiB VRAM"; echo 'vram floor' > REFUSAL; finish 18; }
fi
ram_gb=$(awk '/^MemTotal:/ {printf "%d", $2 * 1024 / 1e9}' /proc/meminfo)
[ "${ram_gb:-0}" -ge "$MIN_RAM_GB" ] || { say "refusing: ${ram_gb} GB host RAM < ${MIN_RAM_GB}"; echo 'host ram floor' > REFUSAL; finish 18; }
disk_gb=$(df -Pk . | awk 'NR == 2 {printf "%d", $4 * 1024 / 1e9}')
[ "${disk_gb:-0}" -ge "$MIN_DISK_GB" ] || { say "refusing: ${disk_gb} GB free disk < ${MIN_DISK_GB}"; echo 'disk floor' > REFUSAL; finish 13; }
printf '{"host_ram_gb": %d, "free_disk_gb": %d, "nproc": %d}\n' "$ram_gb" "$disk_gb" "$(nproc)" > receipts/host.json

unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
VENV="$W/venv-q235"
must create-venv "$CAP_VENV" "$PY" -m venv "$VENV"
export PATH="$VENV/bin:$PATH"
must install "$CAP_INSTALL" python q235_bootstrap.py --cache "$W/.cache/wheels"

DIRECT=(); FLOORS=(--min-h2d-gbs 0 --min-disk-gbs 0 --min-host-copy-gbs 0)
[ "$MODE" = science ] && { DIRECT=(--require-direct)
                           FLOORS=(--min-h2d-gbs "$MIN_H2D_GBS" --min-disk-gbs "$MIN_DISK_GBS" --min-host-copy-gbs "$MIN_HOST_COPY_GBS"); }
phase hostrate "$CAP_HOSTRATE" python q235_hostrate.py --out receipts/hostrate.json --dir "$W" "${FLOORS[@]}" "${DIRECT[@]}"
case "$PHASE_RC" in
  0) ;;
  18) tail -3 logs/hostrate.log; echo 'host rate floor' > REFUSAL; say 'refusing: the link or host-copy rate is below the fit'"'"'s floor'; finish 18 ;;
  13) tail -3 logs/hostrate.log; echo 'disk rate floor' > REFUSAL; say 'refusing: the disk read rate is below the fit'"'"'s floor'; finish 13 ;;
  *)  tail -30 logs/hostrate.log; say "VOID: hostrate (rc $PHASE_RC)"; finish 11 ;;
esac

case "$MODE" in
  science) phase fetch "$CAP_FETCH" python q235_fetch.py --lock model_files.json --dest "$W/hf-cache/model" --out receipts/fetch.json \
             --min-gb-s "$MIN_FETCH_GBS"
           case "$PHASE_RC" in
             0) ;;
             14) tail -3 logs/fetch.log; echo 'pull rate floor' > REFUSAL; say 'refusing: the checkpoint pulls slower than the fit'"'"'s floor'; finish 14 ;;
             *) tail -30 logs/fetch.log; say "VOID: fetch (rc $PHASE_RC)"; finish 11 ;;
           esac
           MODEL="$W/hf-cache/model" DEVICE=cuda ENABLER=fast ;;
  a2000)   must fetch "$CAP_FETCH" python q235_fetch.py --lock model_files.json --dest "$W/hf-cache/model" --slice 4 --slice-dir "$W/hf-cache/slice4" --out receipts/fetch.json
           MODEL="$W/hf-cache/slice4" DEVICE=cuda ENABLER=fast ;;
  cpu)     must fetch "$CAP_FETCH" python q235_fetch.py --lock model_files.json --dest "$W/hf-cache/model" --only 'tokenizer*' special_tokens_map.json --out receipts/fetch.json
           must tiny 300 python r2_tiny.py --out "$W/hf-cache/tiny" --tokenizer-from "$W/hf-cache/model"
           MODEL="$W/hf-cache/tiny" DEVICE=cpu ENABLER=reference ;;
esac
must tokens "$CAP_TOKENS" python q235_tokens.py ds_alpaca.json "$MODEL" --rows "$ROWS" --seq 2048 --out "$W/receipts/tokens.pt" \
  --heldout-out "$W/receipts/heldout.pt"    # Amendment 8: the registered held-out rows, the next ROWS rows of the split

for run in box repeat; do
  extra=(); cap=$CAP_BOX; [ "$run" = repeat ] && { extra=(--repeat --plants); cap=$CAP_REPEAT; }
  phase "$run" "$cap" python q235_box.py "$MODEL" receipts/tokens.pt --device "$DEVICE" --enabler "$ENABLER" --out "receipts/$run" "${extra[@]}"
  case "$PHASE_RC" in
    0) ;;
    3) tail -5 "logs/$run.log"; say "FAIL: a STRICT rule failed in $run (box.json names it)"; finish 12 ;;
    *) tail -30 "logs/$run.log"; say "VOID: $run (rc $PHASE_RC)"; finish 11 ;;
  esac
done
say 'DONE: the step, its repeat and the records; the fidelity read and the verdict are made off the box'
finish 0
