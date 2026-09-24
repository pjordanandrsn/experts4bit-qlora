#!/bin/bash
# bench/p67/p67_run.sh -- lane P67, BOX side (bench/p67/P67-PREREG.md). NOT a training harness: the arms, fixture,
# receipts and TP4 markers are tp4's own (bench/tp4/tp4_run.sh + tp4_arm.py, the machinery P56 drew with). This is
# the registration's guard, run first on the box:
#   1. every staged file matches bench/p67/staged.sha256 (the harness the draw runs is the harness registered);
#   2. the registered knobs are the knobs in force: a value the controller passed that differs is REFUSED, an unset
#      one is SET, and a knob that would change tp4's field fixture must be unset;
#   3. the host's effective RAM (min of MemTotal and the cgroup limit) is at least the Gemma-4 shard class
#      (P56/P55: 96 GiB -- a 49.9 GiB single shard; <= 64 GiB hosts refused or faulted on the map, e4b#344);
# then `exec bash tp4_run.sh`, so the lane's process IS tp4_run.sh (tp4_drive.sh's liveness check greps for it).
# A refusal writes the same markers tp4_run.sh would (nonce, TP4_EXIT_CODE.<nonce>, TP_DONE.<nonce>), so the
# controller reads it as a finished lane with an exit code, never as a hang.
set -uo pipefail
W=${P67_BOX_DIR:-/root/tp4}; cd "$W" || exit 9
say(){ echo "[$(date -u +%FT%TZ)] p67: $*"; }
NONCE=${TP4_RUN_NONCE:?}
refuse(){ local rc=$1; shift; say "REFUSED rc=$rc: $*"; echo "P67_REFUSED rc=$rc $*" >> p67_guard.txt
  printf '%s\n' "$NONCE" > TP4_RUN_NONCE; printf '%s\n' "$rc" > "TP4_EXIT_CODE.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > p67_guard.txt

# 1. the pins
command -v sha256sum >/dev/null || refuse 9 "no sha256sum on the box"
sha256sum -c staged.sha256 > logs_p67_pins.txt 2>&1 || refuse 9 "a staged file differs from bench/p67/staged.sha256 ($(grep -v ': OK$' logs_p67_pins.txt | head -3 | tr '\n' ' '))"
say "pins OK ($(grep -c ': OK$' logs_p67_pins.txt) files)"; echo "PINS OK" >> p67_guard.txt

# 2. the registered knobs (P67-PREREG "The draw"): registered.knobs, pinned by step 1, the same file p67_drive.sh read.
while IFS='|' read -r k v; do
  case "$k" in \#*|"") continue;; esac
  if [ -n "${!k:-}" ] && [ "${!k}" != "$v" ]; then refuse 78 "$k='${!k}' differs from the registered '$v'"; fi
  export "$k=$v"; echo "KNOB $k=$v" >> p67_guard.txt
done < registered.knobs
[ "${TP4_P67:-}" = 1 ] && [ "${TP4_FAMILIES:-}" = gemma4 ] || refuse 78 "registered.knobs did not set the P67 arm set"
# The field fixture is tp4_run.sh's defaults; any of these set would make this a different fixture from the
# standing row and from P56 -- the one comparison the draw exists to make.
for k in TP4_SEQ TP4_MB TP4_ACCUM TP4_R TP4_ALPHA TP4_LR TP4_WD TP4_WARMUP TP4_SCHED TP4_OPTIM TP4_SEED TP4_EVAL_N TP4_EVAL_EVERY TP4_AUTOCAST TP4_DS_ALPACA_SHA; do
  [ -z "${!k:-}" ] || refuse 78 "$k='${!k}' is set; P67 runs tp4's field fixture unchanged"
done
for k in E4B_REFERENCE_EXPERT_ORDER; do   # set per floor arm by tp4_run.sh, never lane-wide
  [ -z "${!k:-}" ] || refuse 78 "$k is set lane-wide; it is a per-arm switch (tp4_run.sh sets it on the perm arms only)"
done

# 3. host RAM against the Gemma-4 shard class
MIN_RAM_GIB=96
MEMINFO=${P67_MEMINFO:-/proc/meminfo}; CGMAX=${P67_CGROUP_MEMMAX:-/sys/fs/cgroup/memory.max}
kb=$(awk '/^MemTotal:/ {print $2}' "$MEMINFO" 2>/dev/null)
case "$kb" in ''|*[!0-9]*) refuse 18 "host RAM unreadable from $MEMINFO";; esac
eff_kb=$kb
cg=$(cat "$CGMAX" 2>/dev/null || true)
case "$cg" in ''|max|*[!0-9]*) ;; *) [ $((cg / 1024)) -lt "$eff_kb" ] && eff_kb=$((cg / 1024));; esac
eff_gib=$((eff_kb / 1048576))
echo "RAM MemTotal_kB=$kb cgroup=${cg:-none} effective_GiB=$eff_gib floor_GiB=$MIN_RAM_GIB" >> p67_guard.txt
# 18: host-limited, but NOT one of the launcher's registered host-limited codes (13/14/17), so the machine is not
# auto-excluded; the receipt names it.
[ "$eff_gib" -ge "$MIN_RAM_GIB" ] || refuse 18 "effective host RAM ${eff_gib} GiB < ${MIN_RAM_GIB} GiB (Gemma-4's 49.9 GiB shard; e4b#344)"
say "RAM OK: ${eff_gib} GiB effective"

echo "GUARD OK" >> p67_guard.txt
[ "${P67_GUARD_ONLY:-0}" = 1 ] && { say "GUARD OK (guard-only)"; exit 0; }
exec bash tp4_run.sh
