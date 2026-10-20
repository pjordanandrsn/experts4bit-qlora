#!/bin/bash
# bench/hostprobe/hostprobe_drive.sh -- the HOST-FACTS PROBE, CONTROLLER side: the launcher's --command
# (bench/hostprobe/PREREG-hostprobe.md). It streams hostprobe_box.py to the box over the launcher's ssh, saves the one
# HOSTPROBE_JSON line to $E4B_RENT_RUN_DIR/hostprobe/facts.json with its log, and exits. No staging, no install, no
# torch, no model: nothing here measures anything. Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [hostprobe_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
RUN_ID=$E4B_RENT_RUN_ID
[[ "$RUN_ID" =~ ^hostprobe-[0-9]+-[0-9]+$ ]] || { say "refusing: run id $RUN_ID is not hostprobe-<machine>-<n>"; exit 78; }
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$E4B_RENT_RUN_DIR/hostprobe"; mkdir -p "$OUT" || exit 78
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -p $E4B_RENT_SSH_PORT root@$E4B_RENT_SSH_HOST"
# one ssh, bounded: the probe reads files and its environment only, so 120 s is generous
perl -e 'alarm 120; exec @ARGV' $SSH "python3 -" < "$HERE/hostprobe_box.py" > "$OUT/probe.log" 2>&1
rc=$?
line=$(grep -m1 '^HOSTPROBE_JSON ' "$OUT/probe.log")
if [ -z "$line" ]; then
  say "no HOSTPROBE_JSON line (ssh rc $rc); see hostprobe/probe.log"; exit 22
fi
printf '%s\n' "${line#HOSTPROBE_JSON }" > "$OUT/facts.json"
say "facts: $(cat "$OUT/facts.json")"
exit 0
