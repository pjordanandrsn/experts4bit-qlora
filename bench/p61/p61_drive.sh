#!/bin/bash
# bench/p61/p61_drive.sh -- lane P61, CONTROLLER side: the launcher's --command (P61-PREREG.md). Stages p61_run.sh,
# the bench (sweep_gemv.py) and P60's recorded routing, starts the runner detached under a fresh nonce, polls
# TP_DONE.<nonce>, fetches receipts. Nothing here creates, destroys or approves compute. Pattern: bench/k18/k18_drive.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p61_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# The runner, the bench and P60's recorded routing are staged; only the served GEMV comes from grouped-nf4-gemm,
# installed on the box at GNF4_SHA.
P60R="$REPO/bench/p60/receipts"
STAGE="$HERE/p61_run.sh $HERE/sweep_gemv.py $HERE/staged.sha256 $P60R/eids_b16.int16.bin $P60R/eids_b16.int16.json"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them, so the check here resolves each
# name to its source and compares hashes rather than running `sha256sum -c` against paths
# that only exist after staging.
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in eids_b16.int16.*) src="$P60R/$name";; *) src="$HERE/$name";; esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the BOX installs must be the e4b this driver ships from -- the launcher has already
# proven that checkout is the manifest's heads.e4b (pod-launch.sh check_tree). A literal default
# here is exactly the unread constant the manifest design exists to remove, and it bit: p39-box1b-3
# installed ef6d532 (pre-#537) while the checkout was 3d22719, so the box rebuilt the same 8-layer
# assignment and VOIDed on its own 48-layer check. Derived, and refused if the tree is dirty.
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
GNF4_SHA=${GNF4_SHA:?the lane measures a gnf4 cut; pass GNF4_SHA=<40-hex> from the manifest}
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P61_POLL_S:-60}; W=/root/p61
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P61_RUN_ID=$RUN_ID P61_RUN_NONCE=$NONCE P61_DEADLINE_EPOCH=$DEADLINE P61_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA "
if [ "${P61_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p61_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p61"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); receipts -> $RUN_DIR/p61; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SSH "cd $W || exit 20; nohup env $PASS bash p61_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P61_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p61" && mkdir -p "$RUN_DIR/p61" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude src --exclude 'venv*' --exclude '.cache' "root@$HOST:$W/" "$RUN_DIR/p61/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p61" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p61/P61_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p61/TP_DONE.$NONCE" ] || { say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p61/P61_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p61/P61_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
