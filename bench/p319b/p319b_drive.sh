#!/bin/bash
# bench/p319b/p319b_drive.sh -- lane P319, CONTROLLER side: the launcher's --command.
# Stages p319b_run.sh + p319b_probe.py, starts the lane detached under a fresh nonce,
# polls for TP_DONE.<nonce> until the deadline, fetches the run dir. One box, no
# arguments; the arms are fixed by bench/p319b/P319-PREREG.md. This lane is SHORT
# (minutes, not hours), so the poll is a plain deadline loop with a liveness probe
# rather than P54's stall model -- a lane that cannot outlive its own clock does
# not need one. Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p319b_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd)
: "${GNF4_BASE_SHA:?set GNF4_BASE_SHA to the RELEASE commit #319 was filed against}"
: "${GNF4_HEAD_SHA:?set GNF4_HEAD_SHA to the branch commit under test}"
for v in GNF4_BASE_SHA GNF4_HEAD_SHA; do
  s=${!v}
  case "$s" in *[!0-9a-f]*|"") say "refusing: $v is not a 40-char hex sha ($s)"; exit 78;; esac
  [ ${#s} -eq 40 ] || { say "refusing: $v is not a 40-char hex sha ($s)"; exit 78; }
done
STAGE="$HERE/p319b_run.sh $HERE/p319b_probe.py"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done

HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR
RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P319B_POLL_S:-20}; W=/root/p319b
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P319B_RUN_ID=$RUN_ID P319B_RUN_NONCE=$NONCE P319B_DEADLINE_EPOCH=$DEADLINE GNF4_BASE_SHA=$GNF4_BASE_SHA GNF4_HEAD_SHA=$GNF4_HEAD_SHA "
if [ "${P319B_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage $STAGE -> root@$HOST:$W ; start: env $PASS bash p319b_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p319b"
  exit 0
fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; base $GNF4_BASE_SHA head $GNF4_HEAD_SHA; receipts -> $RUN_DIR/p319b; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
# The nonce handshake: the child writes the nonce as its first act, so "started"
# means this process started, not that ssh returned.
$SSH "cd $W || exit 20; nohup env $PASS bash p319b_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+60)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P319B_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" \
  || { say "start failed: child did not bind the nonce"; exit 21; }

rc_wait=0
while :; do
  now=$(date -u +%s)
  if [ "$now" -ge "$DEADLINE" ]; then say "deadline reached with no TP_DONE"; rc_wait=24; break; fi
  out=$($SSH "cd $W 2>/dev/null || exit 3; if [ -f TP_DONE.$NONCE ]; then echo DONE:\$(cat P319B_RC 2>/dev/null); else pgrep -f 'bash p319b_run.sh' >/dev/null && echo ALIVE || echo GONE; fi; tail -1 outer.log 2>/dev/null" 2>/dev/null)
  state=$(printf '%s\n' "$out" | head -1)
  last=$(printf '%s\n' "$out" | tail -1)
  case "$state" in
    DONE*) say "lane finished (${state#DONE:}) -- $last"; rc_wait=${state#DONE:}; rc_wait=${rc_wait:-0}; break;;
    ALIVE) say "alive, $(( DEADLINE - now ))s left -- $last";;
    GONE)  say "the lane process is gone with no TP_DONE -- $last"; rc_wait=25; break;;
    *)     say "probe inconclusive ($state) -- retrying";;
  esac
  sleep "$POLL"
done

mkdir -p "$RUN_DIR/p319b"
$SCP -r "root@$HOST:$W/summary.txt" "root@$HOST:$W/logs" "root@$HOST:$W/outer.log" "$RUN_DIR/p319b/" 2>/dev/null \
  || say "warning: some artefacts did not come back"
say "fetched -> $RUN_DIR/p319b (rc=$rc_wait)"
[ -s "$RUN_DIR/p319b/summary.txt" ] && sed -n '1,120p' "$RUN_DIR/p319b/summary.txt"
exit "$rc_wait"
