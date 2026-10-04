#!/bin/bash
# bench/k29/k29_drive.sh -- lane K29, CONTROLLER side: the launcher's --command. Stages k29_run.sh,
# starts it detached under a fresh nonce, polls TP_DONE, fetches receipts. One box, no arguments:
# the probe and sizes are fixed by grouped-nf4-gemm kernel/PREREG-k29-pinned-charge-cgroup-v2.md. Nothing here creates, destroys or approves compute.
# Shape ported from bench/p55/p55_drive.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [k29_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
STAGE="$HERE/k29_run.sh $HERE/pinned_charge_probe.py $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  got=$(sha_of "$HERE/$name")
  [ "$got" = "$want" ] || { say "refusing: $HERE/$name is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the BOX installs must be the e4b this driver ships from. Derived from HEAD, refused
# if the tree is dirty -- a box that installs a commit not describing this tree measures something
# nobody can reconstruct (p39-box1b-3, #537).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${K29_POLL_S:-60}; W=/root/k29
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="K29_RUN_ID=$RUN_ID K29_RUN_NONCE=$NONCE K29_DEADLINE_EPOCH=$DEADLINE K29_INSTANCE_ID=$E4B_RENT_INSTANCE_ID "
if [ "${K29_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash k29_run.sh ; probe (pinned + pageable, registered sizes) ; poll TP_DONE.<nonce> until $DEADLINE ; fetch -> $RUN_DIR/k29"
  echo "DRYRUN e4b=$E4B_SHA repo=$REPO staged=$(echo $STAGE | tr ' ' '\n' | wc -l | tr -d ' ') files, hashes OK"
  exit 0
fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); receipts -> $RUN_DIR/k29; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash k29_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat K29_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
STALL_S=${K29_STALL_S:-900}; MIN_MB=${K29_MIN_PROGRESS_MB:-16}
lane_dead(){ [ "${1:-}" = "0" ] && [ "${2:-}" = "0" ] && echo dead; }   # two DEFINITE zeros
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (a lane whose process is gone for two polls ends the wait rather than running out the clock)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_DU=""; LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*\$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash k29_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | du*}
  duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p'); live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash k29_run.sh' on the box for two consecutive polls and no TP_DONE -- the remote process exited without writing its markers"
    LANE_DEAD=1; break
  fi
  LAST_LIVE=$live
  grew=0; [ -n "$duM" ] && [ -n "$LAST_DU" ] && [ "$duM" -gt "$LAST_DU" ] 2>/dev/null && grew=$((duM - LAST_DU))
  LAST_DU=$duM
  stall=""
  if [ "$grew" -ge "$MIN_MB" ]; then stall=" fetching (+${grew}M)"
  elif [ $((now - LAST_CHANGE)) -ge "$STALL_S" ]; then stall=" STALL? (no change for $((now - LAST_CHANGE))s -- LOOK, do not kill)"; fi
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s$stall"
  sleep "$POLL"
done
rm -rf "$RUN_DIR/k29" && mkdir -p "$RUN_DIR/k29" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" --exclude '.cache' --exclude 'venv*' "root@$HOST:$W/" "$RUN_DIR/k29/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/k29" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/k29/K29_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/k29/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: process gone with no markers -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/k29/K29_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/k29/K29_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
