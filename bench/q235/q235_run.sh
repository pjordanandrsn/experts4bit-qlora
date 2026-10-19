#!/bin/bash
# Q235's box runner (bench/q235/PREREG-q235.md): the registered two-step box on the TC1 nonce/receipt lifecycle -- q235_drive.sh
# starts it through bench/tc1/tc1_drive.sh as TC1_RUNNER. Every phase runs under an alarm capped by the run's deadline less a
# 300 s reserve, and is timed into receipts/phase-timings.jsonl (gate 2's time-left fit reads it). Host floors are refusals with the
# launcher's host codes (18: card, VRAM or host RAM; 13: disk). The fidelity read is made off the box, against the layer-streamed
# bf16 reference (the box ships its per-layer outputs). The checkpoint, the wheel cache and the venv live under paths
# tc1_drive.sh's rsync excludes (hf-cache, .cache, venv*), so only the receipts and logs travel back.
#
# Q235_REHEARSAL=a2000 runs this same script on the lane's RTX A2000 with a depth-truncated slice of the checkpoint (fetched and
# verified like the full one); Q235_REHEARSAL=cpu runs it on a CPU host with a tiny Qwen3-MoE built locally. Neither is ever a
# scientific draw: receipts/run.json says science_eligible false, and the slice and the tiny model are not the registered model.
# Amendment 11: a launcher run id q235-leg-pro6000-N runs the a2000 slice leg on a rented RTX PRO 6000 (mode pro6000), gate 2's
# timing leg; the id is the launcher's registered one, and nothing in the controller's environment reaches this script.
set -uo pipefail
W=${Q235_W:-/root/tc1}; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}
case "$NONCE" in *[!a-zA-Z0-9_-]*|"") exit 78;; esac
[ ${#NONCE} -le 128 ] || exit 78
printf '%s\n' "$NONCE" > TC1_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
say(){ echo "[$(date -u +%FT%TZ)] q235: $*" | tee -a summary.txt; }
mkdir -p logs receipts
: > summary.txt
MODE=${Q235_REHEARSAL:-science}
case "${TC1_RUN_ID:-}" in
  q235-leg-pro6000-[0-9]*) [ -z "${Q235_REHEARSAL:-}" ] || { say "refusing: Q235_REHEARSAL=$Q235_REHEARSAL with leg run id $TC1_RUN_ID"; finish 78; }
                           MODE=pro6000 ;;
esac
case "$MODE" in science|a2000|cpu) ;; pro6000) [ -n "${TC1_RUN_ID:-}" ] || { say 'refusing: pro6000 comes from a leg run id only'; finish 78; } ;;
  *) say "refusing: Q235_REHEARSAL=$MODE"; finish 78;; esac
sha256sum -c q235.sha256 > logs/closure-tripwire.log 2>&1 || { say 'VOID: the staged closure does not match q235.sha256'; finish 9; }
for v in E4B_SHA GNF4_SHA TC1_DEADLINE_EPOCH; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$TC1_DEADLINE_EPOCH" in *[!0-9]*) finish 78;; esac
export E4B_SHA GNF4_SHA TC1_DEADLINE_EPOCH
ELIGIBLE=false; [ "$MODE" = science ] && ELIGIBLE=true
printf '{"schema": "q235-run/1", "mode": "%s", "science_eligible": %s, "e4b_sha": "%s", "gnf4_sha": "%s", "nonce": "%s", "start_utc": "%s"}\n' \
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
# The fixed phases, install (293 s at 806 Mbps on the Amendment 12 leg) and fetch (7,111 s at 806 Mbps; it covers any pull >=
# MIN_FETCH_GBS) keep their sealed caps.
# Box: Amendment 16 sizes it on the box (box_cap below): HALF the time left at the box's start, less the reserve,
# floor((left - 300) / BOX_SHARE). The other half is the repeat's: the repeat is a strict subset of the box's work (one load and about
# five steps, against one load and every box step), so a box inside its half leaves the repeat at least as long. Amendment 12's sealed
# 2,997 s (a warm 4-layer slice x 94/4 x 1.5) is gone: q235-pro6000-7 hit it with 4.9 h left, after a 2,088 s expert fuse.
# If the box hits its half-cap while its per-layer load record (receipts/box/load_layers.jsonl) has no end line, the LOAD ALONE ran
# past the cap: the same code loads the same bytes on every host, so that is host-limited by construction. It exits 18 (a
# registered host floor, which names the machine) as VOID_BY_HOST, not a strike.
# Repeat: Amendment 15 sizes it on the box, from this box's own box phase (repeat_cap below): REPEAT_MARGIN x (the box record's
# load.seconds + REPEAT_STEPS x step.elapsed_s), rounded up, never under 60 s, then bounded by the time left less the reserve like
# every phase. REPEAT_STEPS: after its load, run 2's repeat spent about 139 s, 4.7 of its box's 29.8 s steps (the null, the two
# planted forwards, three sets of layer records and the process start).
# The repeat is a fresh process by design (rule 6), so it reloads the whole model; Amendment 12's leg-scaled 705 s came from a warm
# slice load and could not hold a cold full reload (q235-pro6000-6: 661 s of expert fuse, quantize and offload alone).
CAP_GIT=240 CAP_VENV=120 CAP_INSTALL=1800 CAP_HOSTRATE=600 CAP_FETCH=12000 CAP_TOKENS=300
BOX_SHARE=2 CAP_BOX=""
REPEAT_MARGIN=1.5 REPEAT_STEPS=5 CAP_REPEAT=""
# The host-rate floors (gate 2's premise), from the RTX PRO 6000 leg. Pinned H2D (Amendment 12): max(sealed, the A2000's measured,
# min(28.36, 28.33) / 1.2) = 23.609. The disk (Amendment 12): the leg's box load was warm, so its registered cold-read sample was not
# observed and the floor is max(0.743, 1.145) = 1.145. Host copy (Amendment 13, from the fit): the load's assemble (34.075 s scaled)
# may grow by at most 0.2 x the scaled box (399.5 s), so min(76.64, 70.20) x 34.075 / 433.575 = 5.518 (was 58.5, which refused a
# host whose box fit with ~900 s to spare). The box measures all three (the disk with O_DIRECT) before the checkpoint fetch and
# refuses below any. A rehearsal records only.
MIN_H2D_GBS=23.609 MIN_DISK_GBS=1.145 MIN_HOST_COPY_GBS=5.518
# The pull-rate floor (GB/s): the rate at which the checkpoint, hashed at the rehearsal's rate, still fits CAP_FETCH / 1.5. The fetch
# times its first slice (the small files and 8 shards, ~32 GB) with its own client and refuses (14) below it, before pulling the rest.
MIN_FETCH_GBS=0.062
ROWS=8          # TC1's field-text rows in the one micro-batch (the 30B calibration's and gate 1's); the registering amendment fixes it
PY=python3.11
if [ "$MODE" != science ]; then
  CAP_FETCH=${Q235_CAP_FETCH:-$CAP_FETCH} CAP_BOX=${Q235_CAP_BOX:-} CAP_REPEAT=${Q235_CAP_REPEAT:-}
  PY=${Q235_PYTHON:-$PY}
fi

must require-git "$CAP_GIT" bash dq11_require_git.sh

# Host floors, before any fetch (PREREG "Refusal points" 4-5; the card class is the registering amendment's).
case "$MODE" in
  science) CARD_RE='^NVIDIA RTX PRO 6000'; MIN_VRAM_MIB=94000; MIN_RAM_GB=239; MIN_DISK_GB=550 ;;   # host RAM: Amendment 5
  a2000)   CARD_RE='^NVIDIA RTX A2000 12GB$'; MIN_VRAM_MIB=11000; MIN_RAM_GB=64; MIN_DISK_GB=60 ;;
  pro6000) CARD_RE='^NVIDIA RTX PRO 6000'; MIN_VRAM_MIB=94000; MIN_RAM_GB=239; MIN_DISK_GB=120 ;;   # Amendment 11: a box-class host
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
if [ "$MODE" = pro6000 ]; then
  # Amendment 11, quiet on a rented box: no compute app on the card and utilization.gpu <= 5 % over 10 s, or a refusal. The load
  # average and nproc are recorded, not gated (a container can see its co-tenants' load).
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > logs/quiet-util.log 2>&1 & smi=$!
  sleep 10; kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  umax=$(awk 'BEGIN { m = -1 } /^[ ]*[0-9]+[ ]*$/ { u = $1 + 0; if (u > m) m = u } END { print m }' logs/quiet-util.log)
  printf '{"compute_apps": %d, "utilization_gpu_max_pct": %d, "util_samples": %d, "loadavg": "%s", "nproc": %d}\n' "$apps" "$umax" \
    "$(grep -c '^[ ]*[0-9]' logs/quiet-util.log)" "$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)" "$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)" > receipts/quiet.json
  [ "$apps" -eq 0 ] && [ "$umax" -ge 0 ] && [ "$umax" -le 5 ] \
    || { say "refusing: not quiet ($apps compute apps, utilization.gpu max ${umax} %)"; echo 'not quiet' > REFUSAL; finish 18; }
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

# Amendment 16: what the host-side load runs on, recorded before any timed phase: the CPU model, the logical CPUs, the physical
# cores (unique package/core pairs), threads per core, the affinity, the cgroup quota and torch's default intra-op pool. Recorded,
# never gated here: q235-pro6000-7's fuse ran at torch's default pool of 16 (its physical cores) on 32 logical CPUs.
python - > receipts/cpu.json 2> logs/cpu.log <<'PY' || say 'the host CPU record failed (recorded only, never gated; logs/cpu.log)'
import glob, json, os
import torch
from experts4bit_qlora.util import host_cpu_report
cores = set()
for t in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology"):
    try:
        cores.add((open(t + "/physical_package_id").read().strip(), open(t + "/core_id").read().strip()))
    except OSError:
        pass
model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None)
rep = host_cpu_report()
logical = os.cpu_count()
print(json.dumps({"amendment": 16, "cpu_model": model, "logical_cpus": logical, "physical_cores": len(cores) or None,
                  "threads_per_core": round(logical / len(cores), 2) if cores else None, "affinity_cpus": rep["visible_cpus"],
                  "cpu_quota": rep["cpu_quota"], "quota_detail": rep["quota_detail"], "torch_default_pool": torch.get_num_threads(),
                  "omp_num_threads": rep["omp_num_threads"]}))
PY

DIRECT=(); FLOORS=(--min-h2d-gbs 0 --min-disk-gbs 0 --min-host-copy-gbs 0)
[ "$MODE" = science ] && { DIRECT=(--require-direct)
                           FLOORS=(--min-h2d-gbs "$MIN_H2D_GBS" --min-disk-gbs "$MIN_DISK_GBS" --min-host-copy-gbs "$MIN_HOST_COPY_GBS"); }
[ "$MODE" = pro6000 ] && DIRECT=(--require-direct)        # Amendment 11: the leg records its O_DIRECT read; it refuses nothing by rate
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
  a2000|pro6000)
           must fetch "$CAP_FETCH" python q235_fetch.py --lock model_files.json --dest "$W/hf-cache/model" --slice 4 --slice-dir "$W/hf-cache/slice4" --out receipts/fetch.json
           MODEL="$W/hf-cache/slice4" DEVICE=cuda ENABLER=fast ;;
  cpu)     must fetch "$CAP_FETCH" python q235_fetch.py --lock model_files.json --dest "$W/hf-cache/model" --only 'tokenizer*' vocab.json merges.txt --out receipts/fetch.json
           must tiny 300 python q235_tiny.py --out "$W/hf-cache/tiny" --tokenizer-from "$W/hf-cache/model"
           MODEL="$W/hf-cache/tiny" DEVICE=cpu ENABLER=reference ;;
esac
must tokens "$CAP_TOKENS" python q235_tokens.py ds_alpaca.json "$MODEL" --rows "$ROWS" --seq 2048 --out "$W/receipts/tokens.pt" \
  --heldout-out "$W/receipts/heldout.pt"    # Amendment 8: the registered held-out rows, the next ROWS rows of the split

# --- box_cap (Amendment 16) ---
box_cap(){   # prints the box's cap, half the time left less the reserve, and writes receipts/box-cap.json; non-zero when it cannot
  python - "$TC1_DEADLINE_EPOCH" "$BOX_SHARE" "$CAP_BOX" > receipts/box-cap.json 2> logs/box-cap.log <<'PY' || return 1
import json, sys, time
deadline, share, pinned = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
left = deadline - int(time.time())
derived = (left - 300) // share
if derived <= 0:
    sys.exit(f"{left} s left leaves no box cap")
rec = {"amendment": 16, "left_s": left, "reserve_s": 300, "share": share, "derived_cap_s": derived}
if pinned:                                     # a rehearsal's pinned cap (never a science box): recorded beside the derived cap
    rec["rehearsal_cap_s"] = int(pinned)
print(json.dumps(rec))
PY
  python -c 'import json; r = json.load(open("receipts/box-cap.json")); print(r.get("rehearsal_cap_s", r["derived_cap_s"]))'
}
load_unfinished(){   # 0 when the box's per-layer load record began and never reached its end line (Amendment 16's VOID_BY_HOST)
  [ -s receipts/box/load_layers.jsonl ] && ! grep -q '"done": true' receipts/box/load_layers.jsonl
}
# --- end box_cap ---

# --- repeat_cap (Amendment 15) ---
repeat_cap(){   # prints the repeat's cap from this box's own box record and writes receipts/repeat-cap.json; non-zero when unreadable
  python - "$REPEAT_MARGIN" "$REPEAT_STEPS" "$CAP_REPEAT" > receipts/repeat-cap.json 2> logs/repeat-cap.log <<'PY' || return 1
import json, math, sys
margin, steps, pinned = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
box = json.load(open("receipts/box/box.json"))
load, step = float(box["load"]["seconds"]), float(box["step"]["elapsed_s"])
if not (math.isfinite(load) and math.isfinite(step) and load >= 0 and step > 0):
    sys.exit(f"the box record's load.seconds {load} / step.elapsed_s {step} cannot size the repeat")
rec = {"amendment": 15, "box_load_s": load, "box_step_s": step, "steps_counted": steps, "margin": margin,
       "derived_cap_s": max(math.ceil(margin * (load + steps * step)), 60)}
if pinned:                                     # a rehearsal's pinned cap (never a science box): recorded beside the derived cap
    rec["rehearsal_cap_s"] = int(pinned)
print(json.dumps(rec))
PY
  python -c 'import json; r = json.load(open("receipts/repeat-cap.json")); print(r.get("rehearsal_cap_s", r["derived_cap_s"]))'
}
# --- end repeat_cap ---

for run in box repeat; do
  extra=(--micro-batch 2 --time-moe 8 --time-moe-bound --heldout receipts/heldout.pt)   # box: shape B (Amendment 6), the timed pass (Amendment 7), the held-out read (Amendment 8)
  if [ "$run" = box ]; then
    cap=$(box_cap) && [ -n "$cap" ] || { tail -3 logs/box-cap.log; say 'VOID: the box cap cannot be sized from the time left (Amendment 16)'; finish 11; }
  else
    extra=(--repeat --plants)
    cap=$(repeat_cap) && [ -n "$cap" ] || { tail -3 logs/repeat-cap.log; say 'VOID: the repeat cap cannot be sized from the box record (Amendment 15)'; finish 11; }
  fi
  phase "$run" "$cap" python q235_box.py "$MODEL" receipts/tokens.pt --device "$DEVICE" --enabler "$ENABLER" --out "receipts/$run" "${extra[@]}"
  case "$PHASE_RC" in
    0) ;;
    3) tail -5 "logs/$run.log"; say "FAIL: a STRICT rule failed in $run (box.json names it)"; finish 12 ;;
    *) tail -30 "logs/$run.log"
       if [ "$run" = box ] && [ "$PHASE_RC" = 142 ] && load_unfinished; then   # Amendment 16: the load alone ran past the half-cap
         echo 'VOID_BY_HOST: the load alone ran past the box half-cap' > REFUSAL
         say "VOID_BY_HOST: the box hit its ${cap} s half-cap with the load unfinished ($(grep -c '"layer"' receipts/box/load_layers.jsonl) expert layers done; Amendment 16, not a strike)"
         finish 18
       fi
       say "VOID: $run (rc $PHASE_RC)"; finish 11 ;;
  esac
done
if [ "$MODE" = pro6000 ]; then   # Amendment 11: the second host-rate sample, after the repeat (the fit needs both within 15 %)
  must hostrate2 "$CAP_HOSTRATE" python q235_hostrate.py --out receipts/hostrate2.json --dir "$W" "${FLOORS[@]}" "${DIRECT[@]}"
fi
say 'DONE: the step, its repeat and the records; the fidelity read and the verdict are made off the box'
finish 0
