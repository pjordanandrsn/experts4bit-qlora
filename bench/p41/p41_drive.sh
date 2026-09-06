#!/bin/bash
# bench/p41/p41_drive.sh -- lane P41, controller-side driver: the launcher's --command (experts4bit_qlora/tools/rent.py, e4b#464).
# Reads the box from the launcher's environment (E4B_RENT_SSH_HOST / _PORT, _RUN_DIR, _RUN_ID, _DEADLINE_EPOCH, _WALLCLOCK_S), stages the
# harness (bench/tp3/tp3_arm.py) and p41_run.sh, starts the lane detached on the box, polls TP_DONE while the launcher keeps the
# heartbeat fresh, then rsyncs the arm receipts, stubs, logs, forensics and versions into the run directory. Exit 0 = the lane finished
# (TP_DONE, no BOX_REFUSED); anything else is HARNESS_ERROR in the receipt. Nothing here creates, destroys or approves compute.
#   P41_RATE_USD_H / P41_EST_USD -> the box's STOP-4;  P41_FAMILIES etc. pass through;  P41_DRIVE_DRYRUN=1 prints the commands and exits.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p41_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run this as rent.py --command after a live pre-flight (e4b#464)"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
HARNESS=$REPO/bench/tp3/tp3_arm.py; RUNSH=$HERE/p41_run.sh
[ -s "$HARNESS" ] && [ -s "$RUNSH" ] || { say "refusing: harness or lane script missing ($HARNESS, $RUNSH)"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
POLL=${P41_POLL_S:-60}; W=/root/p41
PASS="P41_RUN_ID=$RUN_ID P41_DEADLINE_EPOCH=$DEADLINE P41_USD_PER_HOUR=${P41_RATE_USD_H:-0} P41_EST_USD=${P41_EST_USD:-0}"
for v in P41_FAMILIES P41_SEQS P41_RANKS P41_PROBE_SEQ P41_STEPS P41_E4B_VER P41_GNF4_VER P41_ANCHOR_STRICT; do [ -n "${!v:-}" ] && PASS="$PASS $v='${!v}'"; done
if [ "${P41_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage: scp -P $PORT $HARNESS $RUNSH root@$HOST:$W/"
  echo "DRYRUN start: $SSH \"cd $W && nohup env $PASS bash p41_run.sh > outer.log 2>&1 &\""
  echo "DRYRUN poll : $SSH test -f $W/TP_DONE  (every $POLL s until $DEADLINE)"
  echo "DRYRUN fetch: rsync -az -e 'ssh -p $PORT' root@$HOST:$W/ $RUN_DIR/p41/  (excluding adapters, e4b-src, data)"
  exit 0
fi
say "run $RUN_ID -> box $HOST:$PORT; receipts -> $RUN_DIR/p41; deadline epoch $DEADLINE"
$SSH "mkdir -p $W/logs && echo '$E4B_RENT_INSTANCE_ID' > $W/INSTANCE_ID" || { say "stage failed: ssh mkdir"; exit 20; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P "$PORT" "$HARNESS" "$RUNSH" "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W && nohup env $PASS bash p41_run.sh > outer.log 2>&1 < /dev/null & echo started" || { say "start failed"; exit 21; }
say "lane started on the box; polling TP_DONE every $POLL s"
LAST=""
while :; do
  now=$(date +%s)
  if $SSH "test -f $W/TP_DONE" 2>/dev/null; then say "TP_DONE seen"; break; fi
  if [ "$now" -ge $((DEADLINE - POLL)) ]; then say "deadline reached without TP_DONE -- fetching what exists"; break; fi
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ "$line" != "$LAST" ] && [ -n "$line" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
mkdir -p "$RUN_DIR/p41"
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude adapters --exclude e4b-src --exclude data --exclude 'venv*' "root@$HOST:$W/" "$RUN_DIR/p41/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p41" | wc -l | tr -d ' ') entries into $RUN_DIR/p41"
[ -f "$RUN_DIR/p41/TP_DONE" ] || { say "lane did not finish (no TP_DONE)"; exit 23; }
[ -f "$RUN_DIR/p41/BOX_REFUSED" ] && { say "box refused by the train anchor (BOX_REFUSED)"; exit 12; }
for m in STOP1 STOP4 STOP5; do [ -f "$RUN_DIR/p41/$m" ] && say "lane stopped by $m (a row, not a failure of the driver)"; done
say "done"; exit 0
