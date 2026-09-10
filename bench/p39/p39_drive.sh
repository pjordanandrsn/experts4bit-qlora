#!/bin/bash
# bench/p39/p39_drive.sh -- lane P39, CONTROLLER side: the launcher's --command (adertha-agents tools/pod-launch.sh ->
# adertha.compute.rent). Reads the box from the launcher's environment (E4B_RENT_SSH_HOST/_PORT, _RUN_DIR, _RUN_ID,
# _DEADLINE_EPOCH, _INSTANCE_ID), stages bo7's pieces + hook v7 + p39_run.sh, starts the lane detached under a fresh
# nonce, polls TP_DONE.<nonce>, fetches receipts (never the 15 GB artifact payloads). Pattern: bench/p41/p41_drive.sh.
#   P39_BOX=1|2 (required); box 2 also needs P39_BOX1_DIR = the fetched box-1 receipt dir (…/p39/box1_out inside it).
# Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p39_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID P39_BOX; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
case "$P39_BOX" in 1|2) ;; *) say "refusing: P39_BOX must be 1 or 2"; exit 78;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE="$HERE/p39_run.sh $HERE/step_decomp.py $HERE/k8_bake.py $HERE/calib.json $HERE/staged.sha256"
for f in $STAGE $HERE/hook/usercustomize.py; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
(cd "$HERE" && sha256sum -c staged.sha256 >/dev/null) || (cd "$HERE" && shasum -a 256 -c staged.sha256 >/dev/null) || { say "refusing: staged pieces differ from staged.sha256"; exit 78; }
E4B_SHA=${E4B_SHA:-ef6d532438e823f29763efb6e65067e02ddbde85}
GNF4_NEW_SHA=${GNF4_NEW_SHA:-00bf78ec07cdfbfd4cef012a2e3b562c0dd71f9b}
GNF4_OLD_SHA=${GNF4_OLD_SHA:-f8f6405d799bbb8efe328a991413cd3869d49bef}
BOX1_FP=""
if [ "$P39_BOX" = 2 ]; then
  [ -n "${P39_BOX1_DIR:-}" ] && [ -s "$P39_BOX1_DIR/box1_out/assignment.json" ] && [ -s "$P39_BOX1_DIR/box1_out/FINGERPRINT" ] \
    || { say "refusing: box 2 needs P39_BOX1_DIR with box1_out/assignment.json and FINGERPRINT (STOP-6)"; exit 78; }
  BOX1_FP=$(tr -d '\n' < "$P39_BOX1_DIR/box1_out/FINGERPRINT"); case "$BOX1_FP" in sha256:*) ;; *) say "refusing: box 1 fingerprint malformed"; exit 78;; esac
fi
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P39_POLL_S:-60}; W=/root/p39
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P39_BOX=$P39_BOX P39_RUN_ID=$RUN_ID P39_RUN_NONCE=$NONCE P39_DEADLINE_EPOCH=$DEADLINE P39_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_NEW_SHA=$GNF4_NEW_SHA GNF4_OLD_SHA=$GNF4_OLD_SHA"
[ -n "$BOX1_FP" ] && PASS="$PASS P39_BOX1_FINGERPRINT=$BOX1_FP"
if [ "${P39_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p39_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p39"; exit 0; fi
say "run $RUN_ID box $P39_BOX nonce=$NONCE -> $HOST:$PORT; receipts -> $RUN_DIR/p39; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook $W/box1" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HERE/hook/usercustomize.py" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
if [ "$P39_BOX" = 2 ]; then $SCP "$P39_BOX1_DIR/box1_out/assignment.json" "$P39_BOX1_DIR/box1_out/manifest.json" "root@$HOST:$W/box1/" || { say "stage failed: box 1 record"; exit 20; }; fi
$SSH "cd $W || exit 20; nohup env $PASS bash p39_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P39_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p39" && mkdir -p "$RUN_DIR/p39" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'artifact*/payloads/layer_*' --exclude work_qwen3 --exclude 'venv*' --exclude '.cache' "root@$HOST:$W/" "$RUN_DIR/p39/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p39" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p39/P39_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p39/TP_DONE.$NONCE" ] || { say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p39/P39_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p39/P39_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "box $P39_BOX complete rc=0"; exit 0
